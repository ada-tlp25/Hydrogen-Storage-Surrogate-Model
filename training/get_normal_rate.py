import pickle
import numpy as np
import pandas as pd
import glob


def convert_normalized_rate(row):
    """
    Convert the normalized net injection/withdrawal value to between -1 and 1 based on the phase of the row.
    This is done so that the model can learn to predict the rate of injection/withdrawal based on the phase of the row.
    If it was kept between 0 and 1, the model wouldn't be able to learn the difference between injection and withdrawal.

    Parameters:
    ----------
    row (pd.Series): A row from the DataFrame containing 'Phase' and 'Normalised_Net_Injection_Withdrawal' columns.

    Returns:
    -------
    float: The converted value between -1 and 1."""
    if row['Phase'] == 'withdrawal':
        return np.float32(row['Normalised_Net_Injection_Withdrawal'] * -1)
    if row['Phase'] == 'injection':
        return np.float32(row['Normalised_Net_Injection_Withdrawal'])
    if row['Phase'] == 'idle':
        return np.float32(0.0)


def get_normal_rate_from_df(sites_directory, initial_synthetic_pkl_path, output_path):
    """
    Add the original LHS rate to the output smry df from eclipse.

    Parameters:
    ----------
    sites_directory (str): The directory containing the individual site pickle files.
    initial_synthetic_pkl_path (str): The path to the initial synthetic pickle file containing the full input DataFrame,
    it contains the normalised injection/withdrawal rates from all the sites at every timestep.
    output_path (str): The path to save the updated individual site pickle files with the added 'Input_Normal_rate' column.

    Returns:
    -------
    None: The function saves the updated individual site pickle files with the added 'Input_Normal_rate' column to the specified output path.
    """
    with open(initial_synthetic_pkl_path, 'rb') as f:
        full_input_df = pd.read_pickle(f)
    full_input_df['Model_rate'] = full_input_df.apply(
        convert_normalized_rate, axis=1)
    # print(full_input_df['Model_rate'])
    print(full_input_df['Site'].value_counts().min())
    indiv_sites_paths = glob.glob(sites_directory + 'Site*pkl')
    # test = [1, 2]
    for i in range(len(indiv_sites_paths)):
        indiv_site = indiv_sites_paths[i]
        with open(indiv_site, 'rb') as f:
            site_dict = pickle.load(f)
        scalar_df = site_dict['Scalars']
        grid_dict = site_dict['Grids']
        grid_keys = list(grid_dict.keys())
        new_grid_dict = {k: grid_dict[k] for k in grid_keys[:120]}
        # print(len(new_grid_dict))
        # print(len(grid_dict))
        new_df = scalar_df[:120].copy()
        # print(len(new_df))
        site = np.int64(indiv_site.split('/')[-1].split('_')[1])
        site_input_df = full_input_df[full_input_df['Site'] == site]
        length = len(new_df)
        # print(
        #   f'Site: {site}, length of scalar df: {length}, length of site_df: {len(site_input_df)}')
        new_df['Period'] = new_df.index.to_period('M')
        new_df.reset_index(drop=True, inplace=True)
        new_df['Input_Normal_rate'] = site_input_df['Model_rate'].values[:length]
        # print(site_input_df.index)
        # print(new_df.index)
        # print(site_input_df['Model_rate'].values[:length])
        # print(new_df['Input_Normal_rate'])
        site_dict['Scalars'] = new_df
        new_site_dict = {'Scalars': new_df, 'Grids': new_grid_dict}
        # print(site_dict['Scalars'].columns)
        # print(site_dict['Scalars']['Input_Normal_rate'])

        with open(output_path + f'Site_{site}_dict.pkl', 'wb') as f:
            pickle.dump(new_site_dict, f)
        perc = ((i + 1) / len(indiv_sites_paths)) * 100
        print(f'Completed {perc:.2f}% of sites')
