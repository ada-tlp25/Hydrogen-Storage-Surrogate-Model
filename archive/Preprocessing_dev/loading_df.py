import pandas as pd
import h5py
import matplotlib.pyplot as plt
import os

# Mapping dictionary created with Anthropic's Claude Opus 4.8 to save time -
# units found from .SMSPEC file and name validated from online ECLIPSE
# documentation.
summary_vectors = {
    'FCGMI_8':    {'name': 'Field component moles in gas phase (immobile) '
                   'component 8 (H2)',  'unit': 'KG-M'},
    'FCGMM_8':    {'name': 'Field component moles in gas phase (mobile),'
                   'component 8 (H2)',          'unit': 'KG-M'},
    'FCMIP_8':    {'name': 'Field component moles in place, component 8 (H2)',
                   'unit': 'KG-M'},
    'FCMIR_8':    {'name': 'Field component mole injection rate, component 8'
                   '(H2)',                     'unit': 'KG-M/D'},
    'FCMPR_8':    {'name': 'Field component mole production rate, component 8'
                   '(H2)',                    'unit': 'KG-M/D'},
    'FCWM_1':     {'name': 'Field component moles in water phase, '
                   'component 1',                         'unit': 'KG-M'},
    'FCWM_2':     {'name': 'Field component moles in water phase,'
                   'component 2',                         'unit': 'KG-M'},
    'FCWM_3':     {'name': 'Field component moles in water phase,'
                   'component 3',                         'unit': 'KG-M'},
    'FCWM_4':     {'name': 'Field component moles in water phase,'
                   'component 4',                         'unit': 'KG-M'},
    'FCWM_5':     {'name': 'Field component moles in water phase,'
                   'component 5',                         'unit': 'KG-M'},
    'FCWM_6':     {'name': 'Field component moles in water phase,'
                   'component 6',                         'unit': 'KG-M'},
    'FCWM_7':     {'name': 'Field component moles in water phase,'
                   'component 7',                         'unit': 'KG-M'},
    'FCWM_8':     {'name': 'Field component moles in water phase,'
                   'component 8 (H2)',                    'unit': 'KG-M'},
    'FGIR':       {'name': 'Field gas injection rate',
                   'unit': 'SM3/DAY'},
    'FGPR':       {'name': 'Field gas production rate',
                   'unit': 'SM3/DAY'},
    'FPR':        {'name': 'Field average reservoir pressure',
                   'unit': 'BARSA'},
    'FPRP':       {'name': 'Field pore-volume-weighted average reservoir '
                   'pressure',                     'unit': 'BARSA'},
    'FYMF_8':     {'name': 'Field gas-phase (vapour) mole fraction, '
                   'component 8 (H2)',                  'unit': 'dimensionless'
                   },
    'TIMESTEP':   {'name': 'Timestep length',
                   'unit': 'DAYS'},
    'WBHP:WELL3': {'name': 'Well bottom-hole pressure, WELL3',
                   'unit': 'BARSA'},
    'YEARS':      {'name': 'Years',
                   'unit': 'YEARS'},
}


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


class eclipse_loader:
    def __init__(self, dir_path, target_length=1002, abbreviate=False):
        self.target_length = target_length
        self.dir_path = dir_path
        self.abbreviate = abbreviate
        file_path_list = []
        site_names = []
        for file in os.listdir(dir_path):
            if file.endswith(".h5"):
                file_path = os.path.join(dir_path, file)
                site_name = int(file.split('_')[0])
                file_path_list.append(file_path)
                site_names.append(site_name)

        full_df = pd.DataFrame()
        for i in range(len(file_path_list)):
            file_path = file_path_list[i]
            with h5py.File(file_path, 'r') as f:
                all_keys = []
                # helped by Claude, to understand how the eclipse files
                # are outputted,
                # specifically the line below which adds all the paths to the
                # datasets.
                f.visit(all_keys.append)
                value_keys = [k for k in all_keys if k.endswith('values')]
                df = pd.DataFrame()
                for key in value_keys:
                    data = f[key][:]
                    column_name = key.split('/')[1]
                    df[column_name] = data
                df['Site'] = site_names[i]
                df['Net Gas Rate (SM3/DAY)'] = df['FGIR'] - df['FGPR']
                df['H2 rate (KG-M/DAY)'] = df['FCMIR_8'] - df['FCMPR_8']
                df['Phase'] = df.apply(determine_results_phase, axis=1)
                df['phase_change'] = (df.groupby('Site')['Phase'].transform(
                    lambda x: x != x.shift()).astype(int))
                df['cycle_id'] = df['phase_change'].cumsum()
                if self.abbreviate == False:
                    df.columns = [summary_vectors.get(col, {}).get(
                        'name', col) for col in df.columns]
                df2 = df[:self.target_length + 1]
                full_df = pd.concat([full_df, df2], ignore_index=True)
        self.df = full_df

    def plot_data(self, columns):
        length = len(columns)
        fig, ax = plt.subplots(length, 1, figsize=(10, 5 * length))
        for idx, column in enumerate(columns):
            ax[idx].plot(self.df['YEARS'], self.df[column], label=column)
            ax[idx].legend()
        plt.tight_layout()
        plt.show()
        return plt.show()
