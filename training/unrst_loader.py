import pickle

import numpy as np
import pandas as pd
import res2df

"""
This function was written with the help of the res2df package built by equinor to access
binary data outputted by the Eclipse Simulations. To help understand how the package works,
Anthropic's Claude Sonnet 5 was used.
The full repository can be found here:
https://github.com/equinor/res2df
"""


columns_to_drop = ['X', 'Y', 'Z', 'Z_MIN',
                   'Z_MAX', 'VOLUME', 'GLOBAL_INDEX',
                   'DATE', 'SGAS', 'YMF8']

static_columns = ['I', 'J', 'K', 'TOPS', 'PERMX',
                  'PERMY', 'PERMZ', 'PORO', 'DZ', 'Years']

dynamic_columns = ['PRESSURE', 'H2 Saturation', 'I', 'J', 'K']


def determine_results_phase(row):
    """
    Determine the phase of a given row based on injection and
    withdrawal values.

    Parameters:
    ----------
    row (pd.Series): A row from the DataFrame containing
    'injection' and 'withdrawal' columns.

    Returns:
    --------
    str: The phase of the row ('injection',
    'withdrawal', or 'idle').
    """
    if row['FGIR'] > 0:
        return 'injection'
    elif row['FGPR'] > 0:
        return 'withdrawal'
    else:
        return 'idle'


def df_to_npy_array(df):
    """
    Convert the Dataframe into a NumPy array of the grid shape so IJK do not need to be stored.
    This saving memory and allows for faster loading of the data.

    Parameters:
    ----------
    df (pd.DataFrame): The DataFrame to convert.

    Returns:
    -------
    np.ndarray: The converted NumPy array in the shape of the grid.
                For this project the output will be 61,61,12,2 where the last dimension is H2 saturation and pressure.
    """
    sorted_df = df.sort_values(by=['I', 'J', 'K']).reset_index(drop=True)
    H2_saturation_array = sorted_df['H2 Saturation'].values.reshape(
        (sorted_df['I'].max(), sorted_df['J'].max(), sorted_df['K'].max())
    )
    Pressure_array = sorted_df['PRESSURE'].values.reshape(
        (sorted_df['I'].max(), sorted_df['J'].max(), sorted_df['K'].max())
    )
    return np.stack((H2_saturation_array, Pressure_array), axis=-1).astype(np.float32)


def convert_UNRST_to_pkl(script_num, local=False, ret_df=False):
    """
    Converts Eclipse Simulation Data into a dictionary with DataFrames for
    each month and a DataFrame for scalars.
    The results are filtered to include only data between January 1991 and
    January 2001 (i.e. 10 years after the pressure ramp up phase)
    The first date is taken from each month to reduce the amount of data
    without losing the overall trend of the simulation.
    The function also calculates the phase of the simulation based on
    injection and withdrawal values.
    """
    DATA_script_file = script_num.split('/')[1]
    if local:
        DATA_script_file = script_num
    resdatafiles = res2df.ResdataFiles(DATA_script_file)
    grid_df = res2df.grid.df(
        resdatafiles,
        rstdates="all",
        vectors=["PRESSURE", "SGAS", "TOPS", "YMF8", 'PERMX', 'PERMY',
                 'PERMZ', 'PORO', 'DZ'],
        stackdates=True)
    grid_df['DATE'] = pd.to_datetime(grid_df['DATE'])
    grid_df['Years'] = (
        (grid_df['DATE'] - grid_df['DATE'].min()).dt.days / 365.25
    ).astype(np.float32)
    grid_df['H2 Saturation'] = (
        grid_df['SGAS'] * grid_df['YMF8']).astype(np.float32)
    # df = grid_df.drop(columns=columns_to_drop)
    grid_df2 = grid_df[(grid_df['DATE'] < '2001-02-01') &
                       (grid_df['DATE'] >= '1991-01-01')]
    month_first = grid_df2.groupby(
        grid_df2['DATE'].dt.to_period('M'))['DATE'].min()
    grid_df2['Years'] = grid_df2['Years'] - 1
    grid_df2 = grid_df2[grid_df2['DATE'].isin(month_first)]
    list3 = month_first.to_list()
    smry = res2df.summary.df(
        resdatafiles, include_restart=True, time_index=list3)
    smry_dropped = smry.drop_duplicates()
    smry_dropped['Phase'] = smry_dropped.apply(determine_results_phase, axis=1)
    smry_dropped['phase_change'] = smry_dropped['Phase'].transform(
        lambda x: x != x.shift()).astype(int)
    smry_dropped['cycle_id'] = smry_dropped['phase_change'].cumsum()
    smry_dropped['Cycle_position'] = smry_dropped.groupby(
        ['cycle_id']).cumcount() + 1
    smry_dropped['Normalised_cycle_pos'] = (
        smry_dropped['Cycle_position'] / smry_dropped.groupby(by=['cycle_id'])
        ['Cycle_position'].transform('max'))
    results_dict = {}
    results_dict['Scalars'] = smry_dropped
    periods = grid_df2['DATE'].dt.to_period('M')
    results_dict['Grids'] = {}
    for i in periods.unique():
        results_dict['Grids'][i] = df_to_npy_array(grid_df2[periods ==
                                                            i][dynamic_columns])
    num = DATA_script_file.split('_')[0]
    if ret_df:
        return results_dict
    else:
        with open(f'Site_{num}_res_dict.pkl', 'wb') as f:
            pickle.dump(results_dict, f)
        return None


def get_static_grid_pkl(script_num, local=False):
    """
    Retrieves the static grid data from the UNRST file and saves it as a
    pickle file.

    Parameters:
    ----------
    script_num (str): The path to the UNRST file.
    local (bool): If True, uses the local path for the UNRST file.

    Returns:
    -------
    None
    """
    DATA_script_file = script_num.split('/')[1]
    if local:
        DATA_script_file = script_num
    resdatafiles = res2df.ResdataFiles(DATA_script_file)
    grid_df = res2df.grid.df(
        resdatafiles,
        rstdates="all",
        vectors=["PRESSURE", "SGAS", "TOPS", "YMF8", 'PERMX', 'PERMY',
                 'PERMZ', 'PORO', 'DZ'],
        stackdates=True)
    print(grid_df.columns)
    grid_df['DATE'] = pd.to_datetime(grid_df['DATE'])
    grid_df['Years'] = (
        (grid_df['DATE'] - grid_df['DATE'].min()).dt.days / 365.25
    ).astype(np.float32)
    grid_df2 = grid_df[(grid_df['DATE'] < '2001-02-01') &
                       (grid_df['DATE'] >= '1991-01-01')]
    month_first = grid_df2.groupby(
        grid_df2['DATE'].dt.to_period('M'))['DATE'].min()
    grid_df2['Years'] = grid_df2['Years'] - 1
    grid_df2 = grid_df2[grid_df2['DATE'].isin(month_first)]
    grid_df2 = grid_df2[static_columns]
    min_years = grid_df2['Years'].min()
    final_df = grid_df2[grid_df2['Years'] == min_years]
    sorted_df = final_df.sort_values(by=['I', 'J', 'K']).reset_index(drop=True)
    tops_array = sorted_df['TOPS'].values.reshape(
        (sorted_df['I'].max(), sorted_df['J'].max(), sorted_df['K'].max())
    )
    permx_array = sorted_df['PERMX'].values.reshape(
        (sorted_df['I'].max(), sorted_df['J'].max(), sorted_df['K'].max())
    )
    permy_array = sorted_df['PERMY'].values.reshape(
        (sorted_df['I'].max(), sorted_df['J'].max(), sorted_df['K'].max())
    )
    permz_array = sorted_df['PERMZ'].values.reshape(
        (sorted_df['I'].max(), sorted_df['J'].max(), sorted_df['K'].max())
    )
    poro_array = sorted_df['PORO'].values.reshape(
        (sorted_df['I'].max(), sorted_df['J'].max(), sorted_df['K'].max())
    )
    dz_array = sorted_df['DZ'].values.reshape(
        (sorted_df['I'].max(), sorted_df['J'].max(), sorted_df['K'].max())
    )
    full_array = np.stack((tops_array, permx_array, permy_array, permz_array,
                           poro_array, dz_array), axis=-1).astype(np.float32)
    if local:
        return full_array
    else:
        with open(f'All_sites_static_grid.pkl', 'wb') as f:
            pickle.dump(full_array, f)
