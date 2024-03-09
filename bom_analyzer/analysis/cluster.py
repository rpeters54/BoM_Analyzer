from umap.umap_ import UMAP
from hdbscan import HDBSCAN
import numpy as np

'''
param_dict is formatted in the same way as the optimizer,
shoud look like this

   param_dict = {
        'min_cluster_size': 48,
        'min_samples': 16,
        'alpha': 0.9615277268640865,
        'n_neighbors': 598,
        'min_dist': 0.9483669074161485
    }

'''


def dimension_reduction(st_data: np.ndarray, param_dict: dict, seed: int) -> np.ndarray:
    """
    Reduces the dimensionality of a NumPy array containing sentence embeddings using UMAP.

    Args:
        st_data (np.ndarray): The NumPy array containing the sentence embeddings (assumed to have higher dimensionality).
        param_dict (dict): A dictionary containing hyperparameters for UMAP, including:
            - n_neighbors: The number of neighbors to consider for each data point.
            - min_dist: The minimum distance between embedded points.
        seed (int): The random seed for UMAP (for reproducibility).

    Returns:
        np.ndarray: The reduced-dimensionality NumPy array representing the data in 2D space.
    """

    return UMAP(n_components=2,
            n_neighbors=param_dict['n_neighbors'],
            random_state=seed,
            min_dist=param_dict['min_dist'],
            n_jobs=1).fit_transform(st_data)


def clustering(umap_data: np.ndarray, param_dict: dict) -> np.ndarray:
    """
    Performs clustering on a 2D NumPy array using HDBSCAN.

    Args:
        umap_data (np.ndarray): The 2D NumPy array containing the data points to cluster.
        param_dict (dict): A dictionary containing hyperparameters for HDBSCAN, including:
            - min_cluster_size: The minimum size of clusters.
            - min_samples: The minimum number of samples required to form a cluster.
            - alpha: The minimum span distance for DBSCAN.

    Returns:
        np.ndarray: A NumPy array containing cluster labels for each data point.
    """

    hdb = HDBSCAN(min_cluster_size=param_dict['min_cluster_size'],
            min_samples=param_dict['min_samples'],
            alpha=param_dict['alpha'],
            gen_min_span_tree=True)

    return hdb.fit_predict(umap_data)
