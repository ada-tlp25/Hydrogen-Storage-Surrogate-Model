import numpy as np
import matplotlib.pyplot as plt


def drop_outliers(data, threshold=2):
    """
    Remove outliers from a list of data based on a specified threshold.

    Parameters:
    ----------
    data (list): A list of numerical data.
    threshold (int): The minimum number of occurrences for a value to be
    retained in the list. Values that occur less than this threshold will be
    removed.

    Returns:
    -------
    np.ndarray: A numpy array containing the data with outliers removed.
    np.ndarray: A numpy array containing the indices of the retained values.
    """
    data = np.asarray(data)
    values, counts = np.unique(data, return_counts=True)
    keep = values[counts >= threshold]
    mask = np.isin(data, keep)
    kept_idxs = np.where(mask)[0]
    return data[mask], kept_idxs


def min_max_normalize(data):
    """
    Normalize a list of data using min-max normalization.

    Parameters:
    ----------
    data (list): A list of numerical data.

    Returns:
    -------
    np.ndarray: A numpy array containing the normalized data.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    return (np.array(data) - min_val) / (max_val - min_val), min_val, max_val


def calculate_cycles(df_all):
    """
    Calculate the injection, withdrawal, and idle cycles for each site in the
    DataFrame.

    Parameters:
    ----------
    df_all (pd.DataFrame): The DataFrame containing all the Natural Gas data

    Returns:
    --------
    inj_dict (dict): A dictionary containing injection cycle lengths and
    amounts.
    with_dict (dict): A dictionary containing withdrawal cycle lengths and
    amounts.
    site_data (dict): A dictionary containing cycle data for each site.
    """
    site_cycle_dict_withdrawal = {}
    site_cycle_dict_injection = {}
    site_cycle_dict_idle = {}
    withdrawal_cycles = []
    injection_cycles = []
    idle_cycles = []
    inj_cycle_amounts_full = []
    with_cycle_amounts_full = []
    inj_rate_individuals = []
    with_rate_individuals = []
    inj_percent_full_all = []
    with_percent_full_all = []
    idle_percent_full_all = []

    for site in df_all['Site'].unique():
        df_site = df_all[df_all['Site'] == site].sort_values(
            'Month').reset_index(drop=True)
        inj_cycle_lengths = []
        with_cycle_lengths = []
        idle_cycle_lengths = []
        inj_cycle_amounts = []
        with_cycle_amounts = []
        inj_percent_full = []
        with_percent_full = []
        idle_percent_full = []
        cycle_count = 0
        cumulative_cycle_amount = 0
        start_phase = None
        current_phase = None
        for row in df_site.itertuples(index=False):
            start_phase = df_site.iloc[0]['Phase']
            if current_phase is None:
                current_phase = start_phase
            if row.Phase == current_phase:
                cycle_count += 1
                cumulative_cycle_amount += row.Net_Injection_Withdrawal_GWh_d
            else:
                if current_phase == 'injection':
                    inj_cycle_lengths.append(cycle_count)
                    inj_cycle_amounts.append(cumulative_cycle_amount)
                    inj_rate_individuals.append(
                        row.Normalised_Net_Injection_Withdrawal)
                    inj_percent_full.append(row.Full_percent)
                elif current_phase == 'withdrawal':
                    with_cycle_lengths.append(cycle_count)
                    with_cycle_amounts.append(cumulative_cycle_amount)
                    with_rate_individuals.append(
                        row.Normalised_Net_Injection_Withdrawal)
                    with_percent_full.append(row.Full_percent)
                else:
                    idle_cycle_lengths.append(cycle_count)
                    idle_percent_full.append(row.Full_percent)
                current_phase = row.Phase
                cycle_count = 1
                cumulative_cycle_amount = row.Net_Injection_Withdrawal_GWh_d

        if current_phase == 'injection':
            inj_cycle_lengths.append(cycle_count)
            inj_cycle_amounts.append(cumulative_cycle_amount)
            inj_rate_individuals.append(
                row.Normalised_Net_Injection_Withdrawal)
            inj_percent_full.append(row.Full_percent)
        elif current_phase == 'withdrawal':
            with_cycle_lengths.append(cycle_count)
            with_cycle_amounts.append(cumulative_cycle_amount)
            with_rate_individuals.append(
                row.Normalised_Net_Injection_Withdrawal)
            with_percent_full.append(row.Full_percent)
        else:
            idle_cycle_lengths.append(cycle_count)
            idle_percent_full.append(row.Full_percent)
        inj_percent_full_all.extend(inj_percent_full)
        with_percent_full_all.extend(with_percent_full)
        idle_percent_full_all.extend(idle_percent_full)
        with_cycle_amounts_full.extend(with_cycle_amounts)
        inj_cycle_amounts_full.extend(inj_cycle_amounts)
        withdrawal_cycles.extend(with_cycle_lengths)
        injection_cycles.extend(inj_cycle_lengths)
        idle_cycles.extend(idle_cycle_lengths)
        site_cycle_dict_withdrawal[site] = [
            with_cycle_lengths, with_cycle_amounts]
        site_cycle_dict_injection[site] = [
            inj_cycle_lengths, inj_cycle_amounts]
        site_cycle_dict_idle[site] = [idle_cycle_lengths]
    # min-max normalisation needed to ensure that the cycle lengths are
    # between 0 and 1 for the synthetic data generation

    injection_cycles, injection_idxs = drop_outliers(injection_cycles, 2)
    withdrawal_cycles, withdrawal_idxs = drop_outliers(withdrawal_cycles, 2)
    idle_cycles, idle_idxs = drop_outliers(idle_cycles, 2)

    injection_cycles2, inj_min, inj_max = min_max_normalize(injection_cycles)
    withdrawal_cycles2, with_min, with_max = min_max_normalize(
        withdrawal_cycles)
    idle_cycles2, idle_min, idle_max = min_max_normalize(idle_cycles)

    inj_cycle_amounts_full = np.asarray(inj_cycle_amounts_full)[injection_idxs]
    with_cycle_amounts_full = np.asarray(
        with_cycle_amounts_full)[withdrawal_idxs]

    inj_rates = np.asarray(inj_rate_individuals)[injection_idxs]
    with_rates = np.asarray(with_rate_individuals)[withdrawal_idxs]

    inj_percent_full = np.asarray(inj_percent_full_all)[injection_idxs]
    with_percent_full = np.asarray(with_percent_full_all)[withdrawal_idxs]
    idle_percent_full = np.asarray(idle_percent_full_all)[idle_idxs]

    inj_dict = {'injection_amounts': inj_cycle_amounts_full,  # cumulative injection amounts for each cycle
                'injection_lengths': injection_cycles,  # lengths of each injection cycle
                'injection_rates': inj_rates,  # individual injection rates for each cycle
                # min and max values for injection cycle lengths
                'injection_min_max': (inj_min, inj_max),
                # normalized injection cycle lengths
                'injection_lengths_normalized': injection_cycles2,
                # percent full values for each injection cycle
                'injection_percent_full': inj_percent_full}

    with_dict = {'withdrawal_amounts': with_cycle_amounts_full,
                 'withdrawal_lengths': withdrawal_cycles,
                 'withdrawal_rates': with_rates,
                 'withdrawal_min_max': (with_min, with_max),
                 'withdrawal_lengths_normalized': withdrawal_cycles2,
                 'withdrawal_percent_full': with_percent_full}
    idle_dict = {'idle_lengths': idle_cycles,
                 'idle_min_max': (idle_min, idle_max),
                 'idle_lengths_normalized': idle_cycles2,
                 'idle_percent_full': idle_percent_full}

    site_data = {'Injection': site_cycle_dict_injection, 'Withdrawal':
                 site_cycle_dict_withdrawal,
                 'Idle': {'idle_lengths': site_cycle_dict_idle}}

    return inj_dict, with_dict, idle_dict, site_data


def country_cycle_stats(df_all):
    """
    Calculate the injection, withdrawal, and idle cycles for each country in
    the DataFrame.

    Parameters:
    ----------
    df_all (pd.DataFrame): The DataFrame containing all the Natural Gas data

    Returns:
    --------
    country_cycle_dict_withdrawal (dict): A dictionary containing withdrawal
    cycle lengths and amounts for each country.
    country_cycle_dict_injection (dict): A dictionary containing injection
    cycle lengths and amounts for each country.
    country_cycle_dict_idle (dict): A dictionary containing idle cycle lengths
    for each country.
    """
    country_cycle_dict_withdrawal = {}
    country_cycle_dict_injection = {}
    country_cycle_dict_idle = {}

    inject_cyles = {}
    withdrawal_cycles = {}
    idle_cycles = {}

    for country in df_all['Country'].unique():
        df_country = df_all[df_all['Country'] == country].sort_values(
            'Month').reset_index(drop=True)
        inj_cycle_lengths = []
        with_cycle_lengths = []
        idle_cycle_lengths = []
        inj_cycle_amounts = []
        with_cycle_amounts = []
        cycle_count = 0
        cumulative_cycle_amount = 0
        start_phase = None
        current_phase = None
        for row in df_country.itertuples(index=False):
            start_phase = df_country.iloc[0]['Phase']
            if current_phase is None:
                current_phase = start_phase
            if row.Phase == current_phase:
                cycle_count += 1
                cumulative_cycle_amount += row.Net_Injection_Withdrawal_GWh_d
            else:
                if current_phase == 'injection':
                    inj_cycle_lengths.append(cycle_count)
                    inj_cycle_amounts.append(cumulative_cycle_amount)
                elif current_phase == 'withdrawal':
                    with_cycle_lengths.append(cycle_count)
                    with_cycle_amounts.append(cumulative_cycle_amount)
                else:
                    idle_cycle_lengths.append(cycle_count)
                current_phase = row.Phase
                cycle_count = 1
                cumulative_cycle_amount = row.Net_Injection_Withdrawal_GWh_d

        if current_phase == 'injection':
            inj_cycle_lengths.append(cycle_count)
            inj_cycle_amounts.append(cumulative_cycle_amount)
        elif current_phase == 'withdrawal':
            with_cycle_lengths.append(cycle_count)
            with_cycle_amounts.append(cumulative_cycle_amount)
        else:
            idle_cycle_lengths.append(cycle_count)

        country_cycle_dict_withdrawal[country] = [
            with_cycle_lengths, with_cycle_amounts]
        country_cycle_dict_injection[country] = [
            inj_cycle_lengths, inj_cycle_amounts]
        country_cycle_dict_idle[country] = [idle_cycle_lengths]
        inject_cyles[country] = len(inj_cycle_lengths)
        withdrawal_cycles[country] = len(with_cycle_lengths)
        idle_cycles[country] = len(idle_cycle_lengths)

    return country_cycle_dict_withdrawal, country_cycle_dict_injection, country_cycle_dict_idle, inject_cyles, withdrawal_cycles, idle_cycles


def plot_cycle_histogram(cycle_lengths, phase, title, xlabel,
                         ylabel="Frequency"):
    """
    Plot a histogram of cycle lengths.

    Parameters:
    ----------
    cycle_lengths (list): A list of cycle lengths to plot.
    title (str): The title of the histogram.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    bins (int): The number of bins for the histogram.

    Returns:
    -------
    None
    """
    if phase == 'idle':
        bins = 5
    else:
        bins = np.linspace(0, 1, 10)
    plt.figure(figsize=(10, 6))
    plt.hist(cycle_lengths, bins=bins,
             edgecolor='black', alpha=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_rate_distribution(df_all, cycle=None, Position=None, All=False,
                           norm=True):
    """
    Plot the distribution of normalised net injection/withdrawal rates for a
    given cycle and position.

    Parameters:
    ----------
    df_all (pd.DataFrame): The DataFrame containing all the Natural Gas data.
    cycle (str): The cycle type to plot ('injection', 'withdrawal', 'idle', or
    None for all cycles).
    Position (int or float): The cycle position to plot (None for all
    positions).
    All (bool): If True, plot distributions for all cycles and positions.
    norm (bool): If True, use normalised cycle positions; otherwise, use raw
    cycle positions.
    """
    if norm:
        grouped_cycle_months = df_all.groupby(
            ['Normalised_cycle_pos_cut', 'Phase']
        )['Normalised_Net_Injection_Withdrawal']
    else:
        grouped_cycle_months = df_all.groupby(['Cycle_position', 'Phase'])[
            'Normalised_Net_Injection_Withdrawal']

    assert cycle in ['injection', 'withdrawal', 'idle',
                     None], "cycle must be either 'injection', "
    "'withdrawal', 'idle' or None"
    assert Position is None or isinstance(
        Position, (int, float)), "Position must be an integer or float or None"

    if All is True:

        for group, data in grouped_cycle_months:
            assert group[1] in ['injection', 'withdrawal',
                                'idle'], """
                                Phase must be either 'injection',
                                'withdrawal' or 'idle'"""
            print(group[0])
            assert group[0] in [
                0.33, 0.66, 1.0], """Normalised cycle position must be either "
                "0.33, 0.66 or 1.0"""
            # don't plot if less than 2 data points as outliers.
            if len(data.index) < 2:
                continue
            plt.figure(figsize=(10, 5))
            plt.hist(data.values, bins=int(np.sqrt(data.index.max())),
                     color='purple', alpha=0.7, edgecolor='black')
            plt.title(f'Cycle Position: {group}')
            plt.xlabel('Normalised Net Injection/Withdrawal')
            plt.ylabel('Frequency')
            plt.show()
    if cycle == 'injection' and Position is None:
        for group, data in grouped_cycle_months:
            if group[1] == 'injection':
                if len(data.index) < 2:
                    continue
                plt.figure(figsize=(10, 5))
                plt.hist(data.values, bins=int(np.sqrt(data.index.max())),
                         color='purple', alpha=0.7, edgecolor='black')
                plt.title(
                    f"""Normalised Injection Rate Distribution for Cycle
                    Position {group}""")
                plt.xlabel('Normalised Net Injection/Withdrawal')
                plt.ylabel('Frequency')
                plt.show()
    if cycle == 'withdrawal' and Position is None:
        for group, data in grouped_cycle_months:
            if group[1] == 'withdrawal':
                if len(data.index) < 2:
                    continue
                plt.figure(figsize=(10, 5))
                plt.hist(data.values, bins=int(np.sqrt(data.index.max())),
                         color='purple', alpha=0.7, edgecolor='black')
                plt.title(f'Normalised Withdrawal Rate Distribution{group}')
                plt.xlabel('Normalised Net Injection/Withdrawal')
                plt.ylabel('Frequency')
                plt.show()
    if cycle == 'idle' and Position is None:
        for group, data in grouped_cycle_months:
            if group[1] == 'idle':
                if len(data.index) < 15:
                    continue
                plt.figure(figsize=(10, 5))
                plt.hist(data.values, bins=int(np.sqrt(data.index.max())),
                         color='purple', alpha=0.7, edgecolor='black')
                plt.title(f'Normalised Idle Rate Distribution{group}')
                plt.xlabel('Normalised Net Injection/Withdrawal')
                plt.ylabel('Frequency')
                plt.show()
    if cycle is not None and Position is not None:
        for group, data in grouped_cycle_months:
            if group[1] == cycle and group[0] == Position:
                if len(data.index) < 15:
                    print(
                        f"""Not enough data for cycle {group[0]} and position
                        {Position}""")
                    continue
                plt.figure(figsize=(10, 5))
                plt.hist(data.values, bins=int(np.sqrt(data.index.max())),
                         color='purple', alpha=0.7, edgecolor='black')
                plt.title(
                    f"""Normalised {group[1].capitalize()} Rate Distribution
                    for Cycle Position {Position}""")
                plt.xlabel('Normalised Net Injection/Withdrawal')
                plt.ylabel('Frequency')
                plt.show()

# def plot_normalised
