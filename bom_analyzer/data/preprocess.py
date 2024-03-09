import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer



def preprocess(
        csv_path: str
) -> np.ndarray:
    """
    Preprocesses a CSV file containing a bill of materials (BOM) for sentence transformation.

    Args:
        csv_path (str): The path to the CSV file.

    Returns:
        np.ndarray: A NumPy array containing the preprocessed BOM data, ready for sentence transformation.
    """

    bom = pd.read_csv(csv_path, header=0, low_memory=False)

    # Replace NaN values with a placeholder string,
    # '#' made sense to me since it doesn't appear anywhere
    # else in the data
    
    if 'HWRMA' in bom.columns:
        # If 'HWRMA' exists, drop it
        bom = bom.fillna("#").drop('HWRMA', axis='columns')

    arr = bom.astype(str).to_numpy()
    return arr


def sentence_transform(
        data: np.ndarray,
        device: str
) -> np.ndarray:
    """
        Encodes a NumPy array of product strings using a sentence transformer model.

        Args:
            data (np.ndarray): A NumPy array of product strings to encode.
            device (str): The device to use for model computation (e.g., 'cpu' or 'cuda').

        Returns:
            np.ndarray: A NumPy array containing the encoded sentence embeddings.
        """

    product_strings = [''.join(row) for row in data]
    model_gte_large = SentenceTransformer('thenlper/gte-large')
    st_data = model_gte_large.encode(product_strings, show_progress_bar=True, device=device)
    return st_data
