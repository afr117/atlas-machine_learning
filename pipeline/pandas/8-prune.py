#!/usr/bin/env python3
"""Module that removes rows with NaN values in the Close column."""


def prune(df):
    """
    Remove any entries where the Close column has NaN values.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with rows containing NaN in Close removed.
    """
    return df.dropna(subset=["Close"])
