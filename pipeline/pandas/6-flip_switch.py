#!/usr/bin/env python3
"""Module that sorts a DataFrame in reverse order and transposes it."""


def flip_switch(df):
    """
    Sort the DataFrame in reverse chronological order and transpose it.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    df = df.sort_index(ascending=False)
    return df.transpose()
