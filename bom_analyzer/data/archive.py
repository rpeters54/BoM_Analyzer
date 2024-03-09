import json
import os
import numpy as np
from typing import *

import pandas as pd


def archive_err_check(archive_path: str) -> None:
    """
    Checks for errors related to the specified archive path.

    Args:
        archive_path (str): The path to the archive file.

    Raises:
        ValueError: If the input archive_path is not a string.
        PermissionError: If there is no write access to the directory for the archive file.
    """
    if not isinstance(archive_path, str):
        raise ValueError("Input 'archive_path' must be a string representing the file path if provided.")
    
    archive_dir = os.path.dirname(archive_path)

    # checks if the path is not a directory (length == 0) or already is a directory
    if len(archive_dir) != 0 and not os.path.isdir(archive_dir):
        os.makedirs(archive_dir)  # Create the directory if it doesn't exist
    
    if len(archive_dir) != 0 and not os.access(archive_dir, os.W_OK):
        raise PermissionError(f"No write access to directory '{archive_dir}' for archive path '{archive_path}'.")
    
    # Create the file if it doesn't exist
    if not os.path.exists(archive_path):
        # Open the file with read and write permissions and create it if it doesn't exist
        os.open(archive_path, os.O_CREAT | os.O_RDWR)


def archive_np_data(
        archive_path: str,
        np_data: np.ndarray
) -> None:
    """
    Saves a NumPy array to a specified archive file.

    Args:
        archive_path (str): The path to the archive file.
        np_data (np.ndarray): The NumPy array to save.

    Raises:
        ValueError: If the input archive_path is not a string.
        FileNotFoundError: If the directory for the archive file does not exist.
        PermissionError: If there is no write access to the directory for the archive file.
    """

    archive_err_check(archive_path)
    np.save(archive_path, np_data)


def archive_pd_data(
        archive_path: str,
        pd_data: pd.DataFrame
) -> None:
    """
    Saves a pandas DataFrame to a specified archive file in CSV format.

    Args:
        archive_path (str): The path to the archive file.
        pd_data (pd.DataFrame): The pandas DataFrame to save.

    Raises:
        ValueError: If the input archive_path is not a string.
        FileNotFoundError: If the directory for the archive file does not exist.
        PermissionError: If there is no write access to the directory for the archive file.
    """

    archive_err_check(archive_path)
    pd_data.to_csv(archive_path, index=False)


def archive_dict(
        archive_path: str,
        dict_data: Dict
) -> None:
    """
    Saves a dictionary to a specified archive file in JSON format.

    Args:
        archive_path (str): The path to the archive file.
        dict_data (Dict): The dictionary to save.

    Raises:
        ValueError: If the input archive_path is not a string.
        FileNotFoundError: If the directory for the archive file does not exist.
        PermissionError: If there is no write access to the directory for the archive file.
    """

    archive_err_check(archive_path)
    with open(archive_path, 'w') as archive_file:
        json.dump(dict_data, archive_file)
