#!/usr/bin/env python3
"""Module compute descriptive statistics for DataFrame."""

import pandas as pd


def analyze(df):
    """
    Compute descriptive statistics for all columns except Timestamp.

    Args:
        df (pd.DataFrame): Input DataFrame containing Timestamp column.

    Returns:
        pd.DataFrame: DataFrame descriptive statistics of numeric columns.
    """
    if "Timestamp" in df.columns:
        return df.drop(columns=["Timestamp"]).describe()
    return df.describe()
