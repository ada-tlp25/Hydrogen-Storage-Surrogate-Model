import numpy as np
import pickle


def get_normalization_stats(train_dataset, output_path=None):
    """
    This function calculates the normalization statistics for the given
    training dataset.
    It computes the minimum and maximum values for the following features:
    - fcmip (Hydrogen Stored)
    - fcmpr (Hydrogen Produced)
    - fcmir (Hydrogen Injected)
    - pressure

    Parameters:
    ----------
    train_dataset (list): A list of the Site directories in the training
    dataset.
    output_path (str, optional): If provided, the normalization statistics
    will be saved

    Returns:
    -------
    normalization_stats (dict): A dictionary containing the min and max values
    for each feature.
    """
    max_fcmip = 0
    max_fcmpr = 0
    max_fcmir = 0

    min_fcmip = 100000000000
    min_fcmpr = 100000000000
    min_fcmir = 100000000000

    min_pressure = 100000000000
    max_pressure = 0

    for i in train_dataset:
        if i.endswith('.json'):
            continue
        else:
            npy_path = "/content/content/drive/MyDrive/tlp25_IRP/npy_files/"
            scalar_arr = np.load(
                npy_path + f"{i}/{i}_scalars.npy", allow_pickle=True)

            min_fcmip = min(min_fcmip, scalar_arr[:, 4].min())
            min_fcmpr = min(min_fcmpr, scalar_arr[:, 3].min())
            min_fcmir = min(min_fcmir, scalar_arr[:, 2].min())
            max_fcmip = max(max_fcmip, scalar_arr[:, 4].max())
            max_fcmpr = max(max_fcmpr, scalar_arr[:, 3].max())
            max_fcmir = max(max_fcmir, scalar_arr[:, 2].max())

            grid_arr = np.load(npy_path + f"{i}/{i}_grids.npy")
            # all periods, all x y z, pressure channel
            pressure_arr = grid_arr[:, :, :, :, 1]
            min_pressure = min(min_pressure, pressure_arr.min())
            max_pressure = max(max_pressure, pressure_arr.max())

        normalization_stats = {
            'min_fcmip': min_fcmip,
            'max_fcmip': max_fcmip,
            'min_fcmpr': min_fcmpr,
            'max_fcmpr': max_fcmpr,
            'min_fcmir': min_fcmir,
            'max_fcmir': max_fcmir,
            'min_pressure': min_pressure,
            'max_pressure': max_pressure
        }
    print(normalization_stats)
    if output_path:
        with open(output_path + 'normalization_stats.pkl', 'wb') as f:
            pickle.dump(normalization_stats, f)
    else:
        return normalization_stats
