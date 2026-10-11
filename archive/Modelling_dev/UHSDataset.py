import pickle
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random


def set_seed(seed=42):
    """
    Set the random seed for reproducibility.

    Parameters:
    ----------
    seed (int): The seed value to set for random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    return True


class UHSDataset(Dataset):
    """
    A custom PyTorch Dataset for handling the UHS output data.

    Parameters:
    ----------
   """

    def __init__(self, file_paths, split='train', transform=None, targets=None, perm=(3, 2, 0, 1)):
        """
        Initialize the dataset.

        Parameters:
        ----------
        file_paths (list): List of paths to the data files.
        split (str): The dataset split to use ('train', 'val', 'test').
        transform (callable, optional): A function/transform to apply
        to the data.
        """
        self.files = file_paths
        self.transform = transform
        self.targets = targets
        self.perm = perm

    def __len__(self):
        # this is the number of input and target pairs in each file
        # 120 months but final month doesn't have a target value so 119 pairs.
        return len(self.files) * 119

    def __getitem__(self, index):
        """
        Retrieve an item from the dataset.

            Parameters:
            ----------
            index (int): The index of the item to retrieve.

            Returns:
            -------
            tuple: A tuple containing the input and target data.
            """
        site_idx = index // 119  # dont need to add 1 as index is 0 based
        month_idx = index - (site_idx * 119)
        with open(self.files[site_idx], 'rb') as f:
            input_dict = pickle.load(f)

        site_dir = f'/content/dataSite_{site_idx}'


        # get scalars df for all tsteps from the site
        scalar_df = input_dict['Scalars']

        # get dict keys so can index by number
        Grid_dict = input_dict['Grids']
        dict_keys = list(Grid_dict.keys())

        input_key = dict_keys[(month_idx)]
        # the target key is the next month
        target_key = dict_keys[(month_idx + 1)]

        target_scalar_row = scalar_df[scalar_df['Period'] == target_key]

        # shape 61 61 12 2
        # channel 1 = H2 Saturation
        # channel 2 = Pressure
        input_grid_arr = Grid_dict[input_key]
        target_grid_arr = Grid_dict[target_key]

        # input rate for the target month
        input_rate = target_scalar_row['Input_Normal_rate']
        rate_arr = np.full((61, 61, 12, 1), input_rate)  # shape 61 61 12 1

        input_arr = np.concatenate(
            (input_grid_arr, rate_arr), axis=3)  # shape 61 61 12 3

        # H2 injection rate after
        target_inj_rate = target_scalar_row['FCMIR_8']
        # H2 production rate after
        target_prod_rate = target_scalar_row['FCMPR_8']
        # H2 moles stored after
        target_moles_stored = target_scalar_row['FCMIP_8']

        # shape 61 61 12 1 UNITS = kg-mole / day
        target_inj_rate_arr = np.full((61, 61, 12, 1), target_inj_rate)
        # shape 61 61 12 1 UNITS = kg-mole / day
        target_prod_rate_arr = np.full((61, 61, 12, 1), target_prod_rate)
        # shape 61 61 12 1 UNITS = kg-mole
        target_moles_stored_arr = np.full((61, 61, 12, 1), target_moles_stored)

        target_arr = np.concatenate(
            # shape 61 61 12 5
            (target_grid_arr, target_inj_rate_arr,
             target_prod_rate_arr, target_moles_stored_arr), axis=3)

        if self.transform is not None:
            input, target = transform_dataset(input_arr, target_arr)

            input_permuted = input.permute(self.perm)
            target_permuted = target.permute(self.perm)
            return input_permuted, target_permuted

        else:
            input_tensor = torch.from_numpy(input_arr)
            target_tensor = torch.from_numpy(target_arr)

            input_tensor = input_tensor.permute(self.perm) # C D H
            target_tensor = target_tensor.permute(
                self.perm)
            return input_tensor, target_tensor


def transform_dataset(input_arr, target_arr, targets=None):
    """
    Transform the input and target arrays into PyTorch tensors.
    The min max values are hard coded due to calculation of the
    min and max values in get_norm_stats.py, this takes a long
    time to run. Due to seed being set, these values will be the
    same for every run. H2 Saturation and Input Rate are not
    normalized as they are already in the range 0-1.
    The Min and Max values for the normalized features are:

    - Pressure: min = 96.261856, max = 104.030624
    - FCMIP_8: min = 15638795, max = 21243478
    - FCMPR_8: min = 0, max = 21173.0625
    - FCMIR_8: min = 0, max = 25313.69921875
    Parameters:
    ----------
    input_arr (np.ndarray): The input array to be transformed.
    target_arr (np.ndarray): The target array to be transformed.
    targets (list, optional): A list of target variables to be transformed.

    Returns:
    -------
    tuple: A tuple containing the transformed input and target tensors.
    """

    H2_saturation = input_arr[:, :, :, 0]
    Rate = input_arr[:, :, :, 2]

    IN_Pressure = input_arr[:, :, :, 1]
    # change these to min and max
    Norm_IN_Pressure = (IN_Pressure - 96.261856) / (104.030624 - 96.261856)

    target_H2_saturation = target_arr[:, :, :, 0]

    target_pressure = target_arr[:, :, :, 1]
    Norm_target_pressure = (target_pressure - 96.261856) / \
        (104.030624 - 96.261856)  # change these to min and max

    target_fcmir = target_arr[:, :, :, 2]
    Norm_target_fcmir = (target_fcmir - 0) / \
        (25313.69921875 - 0)  # change these to min and max

    target_fcmpr = target_arr[:, :, :, 3]
    Norm_target_fcmpr = (target_fcmpr - 0) / \
        (21173.0625 - 0)  # change these to min and max

    target_fcmip = target_arr[:, :, :, 4]
    Norm_target_fcmip = (target_fcmip - 15638795) / \
        (21243478 - 15638795)  # change these to min and max

    Norm_input_arr = np.stack([
        H2_saturation, Norm_IN_Pressure, Rate], axis=-1).astype(np.float32)
    Norm_target_arr = np.stack([target_H2_saturation, Norm_target_pressure,
                                Norm_target_fcmir,
                               Norm_target_fcmpr, Norm_target_fcmip],
                               axis=-1).astype(np.float32)

    input_tensor = torch.from_numpy(Norm_input_arr)
    target_tensor = torch.from_numpy(Norm_target_arr)

    return input_tensor, target_tensor
