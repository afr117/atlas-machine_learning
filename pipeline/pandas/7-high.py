#!/usr/bin/env python3
"""Module that sorts a DataFrame by the High column in descending order."""


def high(df):
    """
    Sort the DataFrame by the High price in descending order.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame sorted by High in descending order.
    """
    return df.sort_values(by="High", ascending=False)
