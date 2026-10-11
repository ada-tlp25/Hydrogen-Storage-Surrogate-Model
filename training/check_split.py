import numpy as np
import pandas as pd


def determine_phase(row):
    if row['Input_Normal_rate'] > 0:
        return 'injection'
    elif row['Input_Normal_rate'] < 0:
        return 'withdrawal'
    else:
        return 'idle'


def get_train_test_distribution(split_files):
    """
    Get the cycle lengths and injection/withdrawal rates for the specified
    files.

    Parameters:
    ----------
    split_files (list): List of paths to the data files in the train/val/test
    split.

    Returns:
    -------
    inj_cycle_length_df (pd.DataFrame): DataFrame containing the cycle lengths
    for each site.
    with_cycle_length_df (pd.DataFrame): DataFrame containing the cycle lengths
    for each site.
    idle_cycle_length_df (pd.DataFrame): DataFrame containing the cycle lengths
    for each site.
    rates_df (pd.DataFrame): DataFrame containing the injection/withdrawal
    rates for each site.
    """

    files_len = len(split_files)
    inj_cycle_length_list = []
    with_cycle_length_list = []
    idle_cycle_length_list = []
    Rates_list = []

    for i in range(files_len):
        site = str(split_files[i].split('/')[-1].split('_')[1])
        scalar_path = split_files[i] + '/Site_' + site + '_scalars.npy'
        scalar_array = np.load(scalar_path, allow_pickle=True)
        file_df = pd.DataFrame(scalar_array)
        file_df.columns = ['Period', 'Input_Normal_rate',
                           'FCMIR_8', 'FCMPR_8', 'FCMIP_8', 'WBHP']
        file_df['phase'] = file_df.apply(determine_phase, axis=1)
        file_df['phase_change'] = file_df['phase'].transform(
            lambda x: x != x.shift()).astype(int)
        file_df['cycle_id'] = file_df['phase_change'].cumsum()
        file_df['Cycle_position'] = file_df.groupby('cycle_id').cumcount() + 1

        inj_df = file_df[file_df['Input_Normal_rate'] > 0]
        with_df = file_df[file_df['Input_Normal_rate'] < 0]
        idle_df = file_df[file_df['Input_Normal_rate'] == 0]
        Inj_Cycle_lengths = pd.DataFrame(inj_df.groupby('cycle_id')[
            'Cycle_position'].max().astype(np.int32))
        with_Cycle_lengths = pd.DataFrame(with_df.groupby('cycle_id')[
            'Cycle_position'].max().astype(np.int32))
        Idle_Cycle_lengths = pd.DataFrame(idle_df.groupby('cycle_id')[
            'Cycle_position'].max().astype(np.int32))
        Inj_Cycle_lengths = Inj_Cycle_lengths.rename(
            columns={'Cycle_position': 'Cycle_lengths'})
        with_Cycle_lengths = with_Cycle_lengths.rename(
            columns={'Cycle_position': 'Cycle_lengths'})
        Idle_Cycle_lengths = Idle_Cycle_lengths.rename(
            columns={'Cycle_position': 'Cycle_lengths'})
        Inj_Cycle_lengths['Site'] = site
        with_Cycle_lengths['Site'] = site
        Idle_Cycle_lengths['Site'] = site
        Normalised_rates = pd.DataFrame(file_df['Input_Normal_rate'])
        Normalised_rates['Site'] = site

        inj_cycle_length_list.append(Inj_Cycle_lengths)
        with_cycle_length_list.append(with_Cycle_lengths)
        idle_cycle_length_list.append(Idle_Cycle_lengths)
        Rates_list.append(Normalised_rates)

    inj_cycle_length_df = pd.concat(
        inj_cycle_length_list, axis=0, ignore_index=True)
    with_cycle_length_df = pd.concat(
        with_cycle_length_list, axis=0, ignore_index=True)
    idle_cycle_length_df = pd.concat(
        idle_cycle_length_list, axis=0, ignore_index=True)
    rates_df = pd.concat(Rates_list, axis=0, ignore_index=True)

    return (inj_cycle_length_df, with_cycle_length_df,
            idle_cycle_length_df, rates_df)
