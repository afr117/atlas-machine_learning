#!/usr/bin/env python3
"""Module concatenates two DataFrames with keys after indexing."""


def concat(df1, df2):
    """
    Index df1 and df2 on Timestamp, select df2 rows up to 1417411920,
    and concatenate df2 above df1 with keys.

    Args:
        df1 (pd.DataFrame): Coinbase DataFrame.
        df2 (pd.DataFrame): Bitstamp DataFrame.

    Returns:
        pd.DataFrame: Concatenated DataFrame with keys 'bitstamp' and 'coinbase'.
    """
    index = __import__('10-index').index
    df1 = index(df1)
    df2 = index(df2)

    df2_selected = df2[df2.index <= 1417411920]

    concatenated = df2_selected.append(df1, keys=["bitstamp", "coinbase"])
    return concatenated
