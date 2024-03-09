import pandas as pd
import numpy as np
from tqdm import trange
from typing import *


# this is a bit of a hacky way to do this, but it works
# given a set of cluster labels, this function outputs all
# the components to a new dataframe
def group_components(
        table: pd.DataFrame,
        labels: np.ndarray
) -> pd.DataFrame:
    """
    Groups components from a DataFrame based on specified cluster labels and extracts relevant data.

    Args:
        table (pd.DataFrame): The input DataFrame containing product data.
        labels (np.ndarray): A NumPy array containing cluster labels to group components by.

    Returns:
        pd.DataFrame: A new DataFrame containing the grouped components with columns:
            - CPN: Component part number
            - DateCode: Manufacturing date code
            - LOTCODE: Lot code
            - MPN: Manufacturer part number
            - RD: Revision date
    """

    # find the index of all columns that contain CPN
    delims = parse_columns(table)
    # create a dataframe to output to
    components = pd.DataFrame(columns=['CPN', 'DateCode', 'LOTCODE', 'MPN', 'RD'])
    # create a copy of the original table that contains only the clusters in labels
    copy = table[table.CLUSTERS.isin(labels)]

    for i in trange(len(delims) - 1):
        temp = pd.DataFrame(columns=['CPN', 'DateCode', 'LOTCODE', 'MPN', 'RD'])
        for j in range(delims[i], delims[i + 1]):
            col = copy.columns[j].partition("_")[0]
            temp[col] = copy.iloc[:, j].values
        temp = temp.dropna(how='all')
        if not temp.empty:
            components = pd.concat([components, temp], ignore_index=True)

    return components


# return a list of the indices of all columns that have a
# title starting with "CPN" and the index of the "HWRMA" column
# used by group_components()
def parse_columns(
        table: pd.DataFrame
) -> List[int]:
    """
    Finds the indices of columns starting with "CPN" and the "HWRMA" column in a DataFrame.
    Helper function used by 'group_components'.

    Args:
        table (pd.DataFrame): The input DataFrame.

    Returns:
        list: A list of column indices, including those starting with "CPN" and the "HWRMA" column.
    """
    cols = table.columns.tolist()
    delims = []
    for i in range(len(cols)):
        if cols[i].startswith("CPN"):
            delims.append(i)
    delims.append(table.columns.get_loc("HWRMA"))
    return delims