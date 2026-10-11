import matplotlib.pyplot as plt


def plot_dataset(input_arr, target_arr, transformed=False, view='top',
                 position=None, index=None, prediction=False):
    """
    Plot the input and target arrays, from a specific view.

    Parameters:
    ----------
    input_arr (np.ndarray): The input array to plot.
    target_arr (np.ndarray): The target array to plot.
    view (str): The view of the plot. Can be 'top', 'North-South', or 'East-
    West'.
    position (tuple): The position of the slice to plot.
    """
    if prediction:
        title_1 = 'Truth'
        title_2 = 'Predicted'

    else:
        title_1 = 'Input'
        title_2 = 'Target'

    if position is None:
        raise ValueError("""Position must be specified to select a slice from
        the dataset.
        - For Top or Bottom View, select between 0(Top) and 11(Bottom)
        - For North-South View, select J between 0(North) and 60(South)
        - For East-West View, select between 0(West) and 60 (East)""")

    if view == 'top':
        fig, ax = plt.subplots(2, 2, figsize=(20, 10))
        k = position

        fig.suptitle(
            f'Top View at K={k} for Sample {index}', fontsize=16)
        H2In = ax[0, 0].imshow(input_arr[0, k, :, :],
                               cmap='viridis', vmin=0, vmax=1)  # input H2
        ax[0, 0].set_title(title_1 + ' H2')
        fig.colorbar(H2In, ax=ax[0, 0])
        if transformed:
            PressIn = ax[1, 0].imshow(
                # input press
                input_arr[1, k, :, :], cmap='viridis', vmin=0, vmax=1)
        else:
            PressIn = ax[1, 0].imshow(
                # input press
                input_arr[1, k, :, :], cmap='viridis', vmin=90, vmax=105)
        ax[1, 0].set_title(title_1 + ' Pressure')
        fig.colorbar(PressIn, ax=ax[1, 0])

        H2Tar = ax[0, 1].imshow(target_arr[0, k, :, :],
                                cmap='viridis', vmin=0, vmax=1)  # target H2
        ax[0, 1].set_title(title_2 + ' H2')
        fig.colorbar(H2Tar, ax=ax[0, 1])

        if transformed:
            PressTar = ax[1, 1].imshow(
                # Target press
                target_arr[1, k, :, :], cmap='viridis', vmin=0, vmax=1)
        else:
            PressTar = ax[1, 1].imshow(
                # Target press
                target_arr[1, k, :, :], cmap='viridis', vmin=90, vmax=105)
        ax[1, 1].set_title(title_2 + ' Pressure')
        fig.colorbar(PressTar, ax=ax[1, 1])
        fig.tight_layout()

    if view == 'North-South':
        fig, ax = plt.subplots(2, 2, figsize=(20, 10))
        j = position

        fig.suptitle(
            f'North-South View at J={j} for Sample {index}', fontsize=16)
        H2In = ax[0, 0].imshow(input_arr[0, :, j, :],
                               cmap='viridis', vmin=0, vmax=1)  # input H2
        ax[0, 0].set_title(title_1 + ' H2')
        fig.colorbar(H2In, ax=ax[0, 0])

        if transformed:
            PressIn = ax[1, 0].imshow(
                # input press
                input_arr[1, :, j, :], cmap='viridis', vmin=0, vmax=1)
        else:
            PressIn = ax[1, 0].imshow(
                # input press
                input_arr[1, :, j, :], cmap='viridis', vmin=90, vmax=105)

        ax[1, 0].set_title(title_1 + ' Pressure')
        fig.colorbar(PressIn, ax=ax[1, 0])

        H2Tar = ax[0, 1].imshow(target_arr[0, :, j, :],
                                cmap='viridis', vmin=0, vmax=1)  # target H2
        ax[0, 1].set_title(title_2 + ' H2')
        fig.colorbar(H2Tar, ax=ax[0, 1])

        if transformed:
            PressTar = ax[1, 1].imshow(
                # Target press
                target_arr[1, :, j, :], cmap='viridis', vmin=0, vmax=1)
        else:
            PressTar = ax[1, 1].imshow(
                # Target press
                target_arr[1, :, j, :], cmap='viridis', vmin=90, vmax=105)
        ax[1, 1].set_title(title_2 + ' Pressure')
        fig.colorbar(PressTar, ax=ax[1, 1])

        fig.tight_layout()

    if view == 'East-West':
        fig, ax = plt.subplots(2, 2, figsize=(20, 10))
        i = position

        fig.suptitle(
            f'East-West View at I={i} for Sample {index}', fontsize=16)

        H2In = ax[0, 0].imshow(input_arr[0, :, :, i],
                               cmap='viridis', vmin=0, vmax=1)  # input H2
        ax[0, 0].set_title(title_1 + ' H2')
        fig.colorbar(H2In, ax=ax[0, 0])

        if transformed:
            PressIn = ax[1, 0].imshow(
                # input press
                input_arr[1, :, :, i], cmap='viridis', vmin=0, vmax=1)
        else:
            PressIn = ax[1, 0].imshow(
                # input press
                input_arr[1, :, :, i], cmap='viridis', vmin=90, vmax=105)
        ax[1, 0].set_title(title_1 + ' Pressure')
        fig.colorbar(PressIn, ax=ax[1, 0])

        H2Tar = ax[0, 1].imshow(target_arr[0, :, :, i],
                                cmap='viridis', vmin=0, vmax=1)  # target H2
        ax[0, 1].set_title(title_2 + ' H2')
        fig.colorbar(H2Tar, ax=ax[0, 1])

        if transformed:
            PressTar = ax[1, 1].imshow(
                # Target press
                target_arr[1, :, :, i], cmap='viridis', vmin=0, vmax=1)
        else:
            PressTar = ax[1, 1].imshow(
                # Target press
                target_arr[1, :, :, i], cmap='viridis', vmin=90, vmax=105)
        ax[1, 1].set_title(title_2 + ' Pressure')
        fig.colorbar(PressTar, ax=ax[1, 1])

        fig.tight_layout()
