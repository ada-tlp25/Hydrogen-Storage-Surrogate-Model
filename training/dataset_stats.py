def get_dataset_stats(df_length, df_rates, name=None):
    """
    Function to print the statistics of the dataset, including cycle lengths
    and rates.
    This was used to check the statistics of the split datasets.

    Parameters:
    df_length (pd.DataFrame): DataFrame containing cycle lengths with a column
    named 'Cycle_lengths'.
    df_rates (pd.DataFrame): DataFrame containing rates with a column named
    'rate'.
    name (str, optional): Name of the dataset. Defaults to None.
    """
    if name is None:
        return print('No name provided')
    print(f"Dataset: {name}")
    print("Cycle Lengths Statistics:")
    print("_________________________________")
    print(f"Number of data points: {len(df_length)}")
    print(f"Mean Cycle Length = {df_length['Cycle_lengths'].mean()}")
    print(f"Median Cycle Length = {df_length['Cycle_lengths'].median()}")
    print(f"Standard Deviation Cycle Length = "
          f"{df_length['Cycle_lengths'].std()}")
    print(f"Mode Cycle Length = {df_length['Cycle_lengths'].mode()[0]}")
    print(f"Minimum Cycle Length = {df_length['Cycle_lengths'].min()}")
    print(f"Maximum Cycle Length = {df_length['Cycle_lengths'].max()}")
    print("Rates Statistics:")
    print("_________________________________")
    print(f"Number of data points: {len(df_rates)}")
    print(f"Mean Rate = {df_rates['rate'].mean()}")
    print(f"Median Rate = {df_rates['rate'].median()}")
    print(f"Standard Deviation Rate = {df_rates['rate'].std()}")
    print(f"Mode Rate = {df_rates['rate'].mode()[0]}")
    print(f"Minimum Rate = {df_rates['rate'].min()}")
    print(f"Maximum Rate = {df_rates['rate'].max()}")
