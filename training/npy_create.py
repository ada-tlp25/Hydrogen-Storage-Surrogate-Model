import pickle
import pandas as pd
import numpy as np
import os


def convert_pkl_to_npy(pkl_directory, npy_directory):
    """
    Converts all the site pickle files in to an individual directory
    for each site with the scalars and grids saved as npy files.
    This is done to save memory and allow for faster loading of the data.
    """
    files = os.listdir(pkl_directory)

    for file in files:
        pkl_path = os.path.join(pkl_directory, file)
        site = file.split('_dict')[0]
        site_path = os.path.join(npy_directory, site)
        if not os.path.exists(site_path):
            os.makedirs(site_path, exist_ok=True)
        else:
            len_files = len(os.listdir(site_path))
            if len_files == 121:
                continue
            else:
                pass

        with open(pkl_path, 'rb') as f:
            full_dict = pickle.load(f)

        scalar_df = full_dict['Scalars']
        grid_dict = full_dict['Grids']

        scalar_df['Period'] = scalar_df['Period'].astype(str)
        relevant_df = scalar_df[[
            'Period', 'Input_Normal_rate', 'FCMIR_8', 'FCMPR_8', 'FCMIP_8',
            'WBHP:WELL3']]
        scalar_npy = relevant_df.to_numpy()
        scalar_npy_path = os.path.join(site_path, site + '_scalars.npy')
        np.save(scalar_npy_path, scalar_npy)

        all_grids = []

        grid_keys = list(grid_dict.keys())
        for key in sorted(grid_keys):
            grid_data = grid_dict[key]
            all_grids.append(grid_data)

        grid_npy = np.stack(all_grids, axis=0)
        np.save(os.path.join(site_path, site + '_grids.npy'), grid_npy)
