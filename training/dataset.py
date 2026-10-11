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
    file_paths (list): List of paths to the train/val/test files.
    transform (callable, optional): A function/transform to apply
    to the data.
    dir (str): The directory containing all the site directories.
    perm (tuple): The permutation order for the input and target tensors.
    norm_stats (dict, optional): A dictionary containing the normalization
    statistics for the dataset.

    Returns:
    -------
    tuple: A tuple containing the transformed input and target tensors.
    The input tensor has the shape (C, D, H, W) and the target tensor has the
    shape (C, D, H, W).
    The input scalars and target scalars are also returned as separate tensors.
    The input scalars contain the following features:
    - inj_rate: Injection rate for the target month.
    - with_rate: Withdrawal rate for the target month.
    - inj_wbhp: Wellhead bottom hole pressure for injection.
    - with_wbhp: Wellhead bottom hole pressure for withdrawal.
    - idle_wbhp: Wellhead bottom hole pressure for idle.
    - month: The month index (0-118). (119 months in total, but the final
    month doesn't have a target value)
    - h2_stored: Hydrogen stored in the reservoir.

    The target scalars contain the following features:
    - target_inj: Target injection rate for the next month.
    - target_prod: Target production rate for the next month.
    - target_moles_stored: Target moles of hydrogen stored in the reservoir
    for the next month.
    """

    def __init__(self, file_paths, transform=None, dir=None,
                 perm=(3, 2, 0, 1), norm_stats=None):
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

        # value between 0 and 118 as zero_index
        month_idx = index - (site_pos * 119)
        if self.dir:
            dir = self.dir
        else:
            dir = (
                "/content/drive/MyDrive/tlp25_IRP/npy_files/")
        site_dir = dir + site_idx + "/"
        scalar_arr_file = f'{site_idx}_scalars.npy'
        scalar_arr = np.load((site_dir + scalar_arr_file), allow_pickle=True)
        # 'Period', 'Input_Normal_rate', 'FCMIR_8', 'FCMPR_8',
        # 'FCMIP_8', 'WBHP:WELL3'
        input_arr_row = scalar_arr[month_idx, :]
        target_arr_row = scalar_arr[month_idx + 1, :]

        grid_file = f'{site_idx}_grids.npy'
        grid_arr = np.load(os.path.join(
            site_dir, grid_file), allow_pickle=True)

        input_grid_arr = grid_arr[month_idx, :, :, :, :]
        target_grid_arr = grid_arr[month_idx + 1, :, :, :, :]

        # input rate for the target month
        input_rate = target_arr_row[1]
        input_wbhp = input_arr_row[5]
        h2_stored = input_arr_row[4]

        if input_rate > 0:
            inj_rate = input_rate
            with_rate = 0
            wbhp_inj = input_wbhp
            wbhp_with = 0
            wbhp_idle = 0

        if input_rate < 0:
            with_rate = input_rate * -1
            inj_rate = 0
            wbhp_with = input_wbhp
            wbhp_inj = 0
            wbhp_idle = 0

        if input_rate == 0:
            inj_rate = 0
            with_rate = 0
            wbhp_inj = 0
            wbhp_with = 0
            wbhp_idle = 0

        input_scalars = np.array(
            [inj_rate, with_rate,  wbhp_inj, wbhp_with, wbhp_idle,
             month_idx, h2_stored])

        # shape 61 61 12 3

        # H2 injection rate after
        target_inj_rate = target_arr_row[2]
        # H2 production rate after
        target_prod_rate = target_arr_row[3]
        # H2 moles stored after
        target_moles_stored = target_arr_row[4]

        target_scalars = np.array(
            [target_inj_rate, target_prod_rate, target_moles_stored])

        if self.transform is not None:
            (input_grid, input_scalars,
             target_grid, target_scalars) = transform_dataset(
                input_grid_arr, input_scalars, target_grid_arr,
                target_scalars, self.norm_stats)

            input_grid_permuted = input_grid.permute(self.perm)
            target_grid_permuted = target_grid.permute(self.perm)

            transformed_inj = target_scalars[0]
            transformed_with = target_scalars[1]
            transformed_inpl = target_scalars[2]
            return (input_grid_permuted, input_scalars, target_grid_permuted,
                    transformed_inj, transformed_with, transformed_inpl)

        else:
            input_tensor = torch.from_numpy(input_grid_arr)
            target_tensor = torch.from_numpy(target_grid_arr)

            input_tensor = input_tensor.permute(self.perm)  # C D H W
            target_tensor = target_tensor.permute(
                self.perm)
            return input_tensor, target_tensor


def transform_dataset(input_arr, input_scalars, target_arr, target_scalars,
                      norm_stats=None):
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
    norm_stats (dict, optional): A dictionary containing the normalization
    statistics.

    Returns:
    -------
    tuple: A tuple containing the transformed input and target tensors.
    """
    # INPUT
    H2_saturation = input_arr[:, :, :, 0]
    IN_Pressure = input_arr[:, :, :, 1]
    # change these to min and max
    Norm_IN_Pressure = (IN_Pressure - norm_stats['min_pressure']) / (
        norm_stats['max_pressure'] - norm_stats['min_pressure'])

    inj_rate = input_scalars[0]
    with_rate = input_scalars[1]
    inj_wbhp = input_scalars[2]
    with_wbhp = input_scalars[3]
    h2_stored = input_scalars[6]

    month = input_scalars[5]

    # These max and min values are hard coded from the ng_loader.py row script
    # generation file. They are the WBHP limits.
    if inj_wbhp > 0:
        Norm_inj_wbhp = (inj_wbhp - 96.5) / (102 - 96.5)
        Norm_with_wbhp = 0
        Norm_idle_wbhp = 0
    elif with_wbhp > 0:
        Norm_with_wbhp = (with_wbhp - 96.5) / (102 - 96.5)
        Norm_inj_wbhp = 0
        Norm_idle_wbhp = 0

    else:
        Norm_inj_wbhp = 0
        Norm_with_wbhp = 0
        Norm_idle_wbhp = 1

    # month is 0-118, as zero index and final month doesn't have a
    # target value.
    norm_month = month / 118

    norm_h2_stored = (h2_stored - norm_stats['min_fcmip']) / \
        (norm_stats['max_fcmip'] - norm_stats['min_fcmip'])

    target_H2_saturation = target_arr[:, :, :, 0]
    target_pressure = target_arr[:, :, :, 1]
    Norm_target_pressure = (target_pressure - norm_stats['min_pressure']) / \
        (norm_stats['max_pressure'] - norm_stats['min_pressure']
         )  # change these to min and max

    Norm_input_arr = np.stack([
        H2_saturation, Norm_IN_Pressure], axis=-1).astype(np.float32)

    Norm_scalar_arr = np.array(
        [inj_rate, with_rate,  Norm_inj_wbhp, Norm_with_wbhp, Norm_idle_wbhp,
         norm_month, norm_h2_stored], dtype=np.float32)

    Norm_target_arr = np.stack([target_H2_saturation, Norm_target_pressure],
                               axis=-1).astype(np.float32)

    target_inj = target_scalars[0]
    target_prod = target_scalars[1]
    target_moles_stored = target_scalars[2]

    norm_inj = (target_inj - norm_stats['min_fcmir']) / \
        (norm_stats['max_fcmir'] - norm_stats['min_fcmir'])
    norm_prod = (target_prod - norm_stats['min_fcmpr']) / \
        (norm_stats['max_fcmpr'] - norm_stats['min_fcmpr'])
    norm_moles_stored = (target_moles_stored - norm_stats['min_fcmip']) / \
        (norm_stats['max_fcmip'] - norm_stats['min_fcmip'])

    norm_target_scalars = np.array(
        [norm_inj, norm_prod, norm_moles_stored], dtype=np.float32)

    input_grid_tensor = torch.from_numpy(Norm_input_arr)
    input_scalar_tensor = torch.from_numpy(Norm_scalar_arr)

    target_grid_tensor = torch.from_numpy(Norm_target_arr)
    target_scalar_tensor = torch.from_numpy(norm_target_scalars)

    return (input_grid_tensor, input_scalar_tensor,
            target_grid_tensor, target_scalar_tensor)
