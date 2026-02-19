#!/usr/bin/env python3
"""Module that slices specific columns and rows from a DataFrame."""


def slice(df):
    """
    Extract the columns High, Low, Close, and Volume_(BTC)
    and select every 60th row from these columns.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The sliced DataFrame.
    """
    return df[["High", "Low", "Close", "Volume_(BTC)"]].iloc[::60]
