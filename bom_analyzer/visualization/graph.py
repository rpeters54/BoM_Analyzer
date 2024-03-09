import matplotlib.pyplot as plt


def plot_data(arr, heading="_"):
    """
    Plots a 2D scatter plot of the given NumPy array with a specified heading.

    Args:
        arr (np.ndarray): The NumPy array containing the data to plot (assumed to have two columns).
        heading (str, optional): The title for the plot. Defaults to "_".
    """

    plt.style.use('dark_background')
    plt.scatter(
        arr[:, 0],
        arr[:, 1],
        )
    plt.title(heading, fontsize=24)
    plt.show()


def plot_labeled_data(arr, labels, heading="", archive_path=None):
    """
    Plots a 2D scatter plot of the given NumPy array with color-coded labels and a specified heading.

    Args:
        arr (np.ndarray): The NumPy array containing the data to plot (assumed to have two columns).
        labels (list): A list of labels corresponding to each data point.
        heading (str, optional): The title for the plot. Defaults to "".
        archive_path (str, optional): The path to save the plot as an image. Defaults to None.
    """

    plt.style.use('dark_background')
    plt.scatter(
        arr[:, 0],
        arr[:, 1],
        c=labels
    )
    plt.title(heading, fontsize=24)
    if archive_path:
        plt.savefig(archive_path)
    plt.show()

