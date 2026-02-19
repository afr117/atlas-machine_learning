#!/usr/bin/env python3
"""Module that converts selected DataFrame columns to a NumPy array."""


def array(df):
    """
    Select the last 10 rows of the High and Close columns and
    convert them to a numpy.ndarray.

    Args:
        df (pd.DataFrame): DataFrame containing 'High' and 'Close' columns.

    Returns:
        numpy.ndarray: The resulting array of the last 10 rows
        of High and Close values.
    """
    return df[["High", "Close"]].tail(10).to_numpy()
