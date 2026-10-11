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

    def __init__(self, file_paths, split='train', transform=None, dir=None, perm=(3, 2, 0, 1), norm_stats=None):
        """
        Initialize the dataset.

        Parameters:
        ----------
        file_paths (list): List of paths to the data files.
        split (str): The dataset split to use ('train', 'val', 'test').
        transform (callable, optional): A function/transform to apply
        to the data.
        dir (str): The directory containing the data files.
        """
        self.files = file_paths
        self.transform = transform
        self.dir = dir
        self.perm = perm
        self.norm_stats = norm_stats

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
        site_pos = index // 119
        site_idx = self.files[site_pos]  # 0, 1 , 2 , 3

        month_idx = index - (site_pos * 119)
        if self.dir:
            dir = self.dir
        else:
            dir = (
                f"/content/drive/MyDrive/tlp25_IRP/npy_files/")
        site_dir = dir + site_idx + "/"
        scalar_arr_file = f'{site_idx}_scalars.npy'
        scalar_arr = np.load((site_dir + scalar_arr_file), allow_pickle=True)
        # 'Period', 'Input_Normal_rate', 'FCMIR_8', 'FCMPR_8', 'FCMIP_8', 'WBHP:WELL3'
        input_arr_row = scalar_arr[month_idx, :]
        target_arr_row = scalar_arr[month_idx + 1, :]

        grid_file = f'{site_idx}_grids.npy'
        grid_arr = np.load(os.path.join(
            site_dir, grid_file), allow_pickle=True)

        input_grid_arr = grid_arr[month_idx, :, :, :, :]
        target_grid_arr = grid_arr[month_idx + 1, :, :, :, :]

        # input rate for the target month
        input_rate = target_arr_row[1]
        rate_arr = np.full((61, 61, 12, 1), input_rate)  # shape 61 61 12 1

        input_arr = np.concatenate(
            (input_grid_arr, rate_arr), axis=3)  # shape 61 61 12 3

        # H2 injection rate after
        target_inj_rate = target_arr_row[2]
        # H2 production rate after
        target_prod_rate = target_arr_row[3]
        # H2 moles stored after
        target_moles_stored = target_arr_row[4]

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
            input, target = transform_dataset(
                input_arr, target_arr, self.norm_stats)

            input_permuted = input.permute(self.perm)
            target_permuted = target.permute(self.perm)
            return input_permuted, target_permuted

        else:
            input_tensor = torch.from_numpy(input_arr)
            target_tensor = torch.from_numpy(target_arr)

            input_tensor = input_tensor.permute(self.perm)  # C D H W
            target_tensor = target_tensor.permute(
                self.perm)
            return input_tensor, target_tensor


def transform_dataset(input_arr, target_arr, norm_stats=None):
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
    norm_stats (dict, optional): A dictionary containing the normalization statistics.

    Returns:
    -------
    tuple: A tuple containing the transformed input and target tensors.
    """

    H2_saturation = input_arr[:, :, :, 0]
    Rate = input_arr[:, :, :, 2]

    IN_Pressure = input_arr[:, :, :, 1]
    # change these to min and max
    Norm_IN_Pressure = (IN_Pressure - norm_stats['min_pressure']) / (
        norm_stats['max_pressure'] - norm_stats['min_pressure'])

    target_H2_saturation = target_arr[:, :, :, 0]

    target_pressure = target_arr[:, :, :, 1]
    Norm_target_pressure = (target_pressure - norm_stats['min_pressure']) / \
        (norm_stats['max_pressure'] - norm_stats['min_pressure']
         )  # change these to min and max

    target_fcmir = target_arr[:, :, :, 2]
    Norm_target_fcmir = (target_fcmir - norm_stats['min_fcmir']) / \
        (norm_stats['max_fcmir'] - norm_stats['min_fcmir']
         )  # change these to min and max

    target_fcmpr = target_arr[:, :, :, 3]
    Norm_target_fcmpr = (target_fcmpr - norm_stats['min_fcmpr']) / \
        (norm_stats['max_fcmpr'] - norm_stats['min_fcmpr']
         )  # change these to min and max

    target_fcmip = target_arr[:, :, :, 4]
    Norm_target_fcmip = (target_fcmip - norm_stats['min_fcmip']) / \
        (norm_stats['max_fcmip'] - norm_stats['min_fcmip']
         )  # change these to min and max

    Norm_input_arr = np.stack([
        H2_saturation, Norm_IN_Pressure, Rate], axis=-1).astype(np.float32)
    Norm_target_arr = np.stack([target_H2_saturation, Norm_target_pressure,
                                Norm_target_fcmir,
                               Norm_target_fcmpr, Norm_target_fcmip],
                               axis=-1).astype(np.float32)

    input_tensor = torch.from_numpy(Norm_input_arr)
    target_tensor = torch.from_numpy(Norm_target_arr)

    return input_tensor, target_tensor
