import pandas as pd
import os

col_names = {'Status',
             'Gas Day Start',
             'Gas Day End',
             'Gas in storage (TWh)',
             'Full (%)',
             'Trend (%)',
             'Injection (GWh/d)',
             'Withdrawal (GWh/d)',
             'Technical Capacity (TWh)',
             'Injection capacity (GWh/d)',
             'Withdrawal capacity (GWh/d)',
             'Contracted Capacity (TWh)',
             'Available Capacity (TWh)'}

column_mapping = {
    'status': 'Status',
    'gasDayStart': 'Gas_Day_Start',
    'gasDayEnd': 'Gas_Day_End',
    'gasInStorage': 'Gas_in_storage_TWh',
    'full': 'Full_percent',
    'trend': 'Trend_percent',
    'injection': 'Injection_GWh_d',
    'withdrawal': 'Withdrawal_GWh_d',
    'Net': 'Net_Injection_Withdrawal_GWh_d',
    'Normalised Net': 'Normalised_Net_Injection_Withdrawal',
    'phase': 'Phase',
    'phase_change': 'Phase_Change',
    'cycle_id': 'Cycle_ID',
    'workingGasVolume': 'Technical_Capacity_TWh',
    'injectionCapacity': 'Injection_capacity_GWh/d',
    'withdrawalCapacity': 'Withdrawal_capacity_GWh_d',
    'contractedCapacity': 'Contracted_Capacity_TWh',
    'availableCapacity': 'Available_Capacity_TWh'
}

well = "WELL3"
stream = "WSTR1"


def generate_row_script(row, MAXrate=6*(10**5), BHP_MAX=102, BHP_MIN=96.5,
                        synthetic=False,
                        time_unit='Month'):
    """
    Generate an Eclipse script based on the data in a given row of the
    DataFrame.

    Parameters:
    ----------
    row (pd.Series): A row from the DataFrame containing the necessary data.
    MAXrate (float): The maximum rate for injection/withdrawal, used to
    calculate the actual rate based on the normalized net injection/withdrawal.
    BHP_MAX (float): The maximum bottom hole pressure for the well.
    BHP_MIN (float): The minimum bottom hole pressure for the well.
    Synthetic (bool): Needed when generating synthetic data, as the timestep
    is different for synthetic data.

    Returns:
    -------
    str: The generated Eclipse script for the given row.
    """
    rate = row['Normalised_Net_Injection_Withdrawal'] * MAXrate
    List = []
    if row['Phase'] == 'injection':
        List += ["WCONINJE",
                 f" {well} GAS 1* RATE {rate} 1* {BHP_MAX}/", "    /"]
        if row['Phase_Change'] == 1:
            List += ["WELLOPEN", f"'{well}' /", " /"]
            List += ["WINJGAS", f"'{well}' STREAM '{stream}' 2* /", " /"]
    elif row['Phase'] == 'withdrawal':
        # List += ["WCONPROD", f" {well} OPEN WGRA 9* {rate} /", " /"]
        List += ["WCONPROD",
                 f" {well} OPEN WGRA 5* {BHP_MIN} 3* {rate} /", " /"]
        if row['Phase_Change'] == 1:
            List += ["WELLOPEN", f"'{well}' /", " /"]

    elif row['Phase'] == 'idle':
        List += ["WELLSHUT", f"'{well}' /", " /"]

    else:
        raise ValueError(f"Unknown phase: {row['Phase']}")
    if time_unit == 'months':
        List += ["TSTEP", "30 /", ""]

    if time_unit == 'weeks':
        List += ["TSTEP", "7 /", ""]
    elif time_unit == 'days':
        List += ["TSTEP", "1 /", ""]

    return "\n".join(List)


def determine_phase(row):
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
    if row['injection'] > row['withdrawal']:
        return 'injection'
    elif row['withdrawal'] > row['injection']:
        return 'withdrawal'
    else:
        return 'idle'


def normalise_rate(df, row):
    """
    Normalise the rate of a given row based on the maximum and minimum
    injection/withdrawal values for the site.

    Parameters:
    ----------
    df (pd.DataFrame): The DataFrame containing all the data.
    row (pd.Series): A row from the DataFrame containing
    'injection' and 'withdrawal' columns.

    Returns:
    --------
    float: The normalised rate of the row.
    """
    df_site = df[df['Site'] == row['Site']]

    if row['phase'] == 'injection':
        return (row['injection'] - df_site['injection'].min()) / (
            df_site['injection'].max() - df_site['injection'].min())
    elif row['phase'] == 'withdrawal':
        return (row['withdrawal'] - df_site['withdrawal'].min()) / (
            df_site['withdrawal'].max() - df_site['withdrawal'].min())
    elif row['phase'] == 'idle':
        return 0
    else:
        raise ValueError(f"Unknown phase: {row['phase']}")


def df_all_cycles(directory_path, MAXrate=6*(10**5), time_unit='days',
                  synthetic=False):
    """
    Load all cycles from the given directory path and return a DataFrame with
    the combined data.

    Parameters:
    ----------
    directory_path (str): The path to the directory containing the JSON files.
    MAXrate (float): The maximum rate for injection/withdrawal, used to
    calculate the actual rate based on the normalized net injection/withdrawal.
    time_unit (str): The unit of time for the duration calculation ('days',
    'weeks' or 'months').
    synthetic (bool): Whether to generate synthetic data or not.

    Returns:
    -------
    pd.DataFrame: A DataFrame containing the combined data from all
    JSON files in the directory and additional calculated columns.
    """
    files = os.listdir(directory_path)
    individual_files = [f for f in files if f.endswith('.json')]
    df = pd.DataFrame()
    for file in individual_files:
        df_temp = pd.read_json(os.path.join(directory_path, file))
        country = file.split('_')[1]
        site = file.split('_')[0]
        if 'Post-Brexit' in site:
            site = site.replace('(Post-Brexit)', '').strip()
        df_temp['Site'] = site
        df_temp['Country'] = country.split('.')[0]
        df_temp.drop(df_temp[(df_temp['status'] == 'N')].index, inplace=True)
        numeric_cols = df_temp.select_dtypes(include=['number']).columns
        mask = df_temp[numeric_cols].eq(0).sum(
            axis=1) / len(numeric_cols) > 0.5
        df_temp.drop(df_temp[mask].index, inplace=True)
        df = pd.concat([df, df_temp], ignore_index=True)

    mask = (df['injection'] > df['injectionCapacity']) | (
        df['withdrawal'] > df['withdrawalCapacity'])  # remove invalid data
    df = df[~mask]
    # negative indicates withdrawal
    df['gasDayStart'] = pd.to_datetime(df['gasDayStart'])
    df = df.sort_values(['Country', 'Site', 'gasDayStart']
                        ).reset_index(drop=True)
    df['Month'] = df['gasDayStart'].dt.to_period('M')
    df['Week'] = df['gasDayStart'].dt.to_period('W')
    if time_unit == 'months':
        df_monthly = (
            df.groupby(["Country", "Site", "Month"], as_index=False)
            .agg(injection=("injection", "mean"),
                 withdrawal=("withdrawal", "mean"),
                 gasInStorage=("gasInStorage", "last"),
                 full=("full", "last"),
                 workingGasVolume=("workingGasVolume", "last"),
                 injectionCapacity=("injectionCapacity", "last"),
                 withdrawalCapacity=("withdrawalCapacity", "last"),
                 contractedCapacity=("contractedCapacity", "last"),
                 availableCapacity=("availableCapacity", "last")
                 )
        )
        df = df_monthly
    elif time_unit == 'weeks':
        df_weekly = (
            df.groupby(["Country", "Site", 'Week'], as_index=False)
            .agg(injection=("injection", "mean"),
                 withdrawal=("withdrawal", "mean"),
                 gasInStorage=("gasInStorage", "last"),
                 full=("full", "last"),
                 workingGasVolume=("workingGasVolume", "last"),
                 injectionCapacity=("injectionCapacity", "last"),
                 withdrawalCapacity=("withdrawalCapacity", "last"),
                 contractedCapacity=("contractedCapacity", "last"),
                 availableCapacity=("availableCapacity", "last")
                 )
        )
        df = df_weekly

    elif time_unit == 'days':
        pass  # already days

    else:
        raise ValueError(
            "time_unit must be either 'days', 'weeks', or 'months'")
    df['phase'] = df.apply(determine_phase, axis=1)
    # df['Normalised Net'] = df.apply(
    #    lambda row: normalise_rate(df, row), axis=1)
    df['Net'] = df['injection'] - df['withdrawal']
    df['Normalised Net'] = df.groupby(['Country', 'Site']).apply(lambda x: (
        x['Net'] - x['Net'].min()) / (x['Net'].max() - x['Net'].min()),
        include_groups=False
    ).reset_index(level=[0, 1, 2], drop=True)
    df['phase_change'] = (df.groupby('Site')['phase'].transform(
        lambda x: x != x.shift()).astype(int))
    df['cycle_id'] = df['phase_change'].cumsum()
    df.columns = [column_mapping.get(col, col) for col in df.columns]
    if time_unit == 'days':
        df.sort_values(['Site', 'Gas_Day_Start']).reset_index(
            drop=True, inplace=True)
    elif time_unit == 'weeks':
        df.sort_values(['Site', 'Week']).reset_index(drop=True, inplace=True)
    else:
        df.sort_values(['Site', 'Month']).reset_index(drop=True, inplace=True)
    df['Row_Script'] = df.apply(
        generate_row_script, axis=1, MAXrate=MAXrate, time_unit=time_unit,
        synthetic=synthetic)
    df['Cycle_position'] = df.groupby(['Site', 'Cycle_ID']).cumcount() + 1
    df['Normalised_cycle_pos'] = (
        df['Cycle_position'] / df.groupby(by=['Site', 'Cycle_ID'])
        ['Cycle_position'].transform('max'))
    df['Normalised_cycle_pos_cut'] = pd.cut(
        df['Normalised_cycle_pos'],
        bins=[0.0, 0.33, 0.66, 1.0],
        labels=[0.33, 0.66, 1.0]
    )
    return df


def cycles(directory_path, MAXrate=6*(10**5), time_unit='days',
           synthetic=False):
    """
    Load all cycles from the given directory path and return a DataFrame with
    the combined data.

    Parameters:
    ----------
    directory_path (str): The path to the directory containing the JSON files.
    MAXrate (float): The maximum rate for injection/withdrawal, used to
    calculate the actual rate based on the normalized net injection/withdrawal.
    time_unit (str): The unit of time for the duration calculation ('days',
    'weeks' or 'months').
    Synthetic (bool): Whether to generate synthetic data or not.
    Returns:
    --------
    pd.DataFrame: A DataFrame containing the combined data from all JSON
    files in the directory and additional calculated columns.
    """
    df = df_all_cycles(directory_path, MAXrate=MAXrate,
                       time_unit=time_unit, synthetic=synthetic)
    cycles = df.groupby(['Site', 'cycle_id']).agg(
        start_date=('gasDayStart', 'first'),
        end_date=('gasDayEnd', 'last'),
        phase=('phase', 'first'),
        duration=('gasDayStart', lambda x: (x.max() - x.min()).days + 1),
        total_injection=('injection', 'sum'),
        total_withdrawal=('withdrawal', 'sum'),
        net_change=('Net', 'sum')
    ).reset_index()
    return cycles
