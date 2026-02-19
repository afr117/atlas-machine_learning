#!/usr/bin/env python3
"""Module that creates a pandas DataFrame from a NumPy ndarray."""

import pandas as pd


def from_numpy(array):
    """
    Create a pandas DataFrame from a NumPy ndarray.

    The columns of the DataFrame are labeled in alphabetical
    order and capitalized (A, B, C, ...).

    Args:
        array (np.ndarray): The NumPy array to convert.

    Returns:
        pd.DataFrame: The resulting pandas DataFrame.
    """
    num_cols = array.shape[1]
    columns = [chr(ord('A') + i) for i in range(num_cols)]
    return pd.DataFrame(array, columns=columns)
