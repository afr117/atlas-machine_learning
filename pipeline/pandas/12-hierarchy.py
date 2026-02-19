#!/usr/bin/env python3
"""Module creates hierarchical concatenation of two DataFrames."""

import pandas as pd


def hierarchy(df1, df2):
    """
    Concatenate df2 and df1 in a timestamp range with a hierarchical index.

    Args:
        df1 (pd.DataFrame): Coinbase DataFrame.
        df2 (pd.DataFrame): Bitstamp DataFrame.

    Returns:
        pd.DataFrame: Concatenated DataFrame MultiIndex (Timestamp, source).
    """
    index = __import__('10-index').index
    df1 = index(df1)
    df2 = index(df2)

    # Select the timestamp range
    df1_sel = df1[(df1.index >= 1417411980) & (df1.index <= 1417417980)]
    df2_sel = df2[(df2.index >= 1417411980) & (df2.index <= 1417417980)]

    # Concatenate with keys
    concatenated = pd.concat([df2_sel, df1_sel], keys=["bitstamp", "coinbase"])

    # Swap levels so Timestamp is first
    concatenated = concatenated.swaplevel(0, 1).sort_index()

    return concatenated

