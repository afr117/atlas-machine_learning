#!/usr/bin/env python3
"""Module computes descriptive statistics for DataFrame without importing."""


def analyze(df):
    """
    Compute descriptive statistics for all columns except Timestamp.

    Args:
        df (pd.DataFrame): Input DataFrame containing Timestamp column.

    Returns:
        pd.DataFrame: DataFrame containing descriptive statistics numeric columns.
    """
    if "Timestamp" in df.columns:
        return df.drop(columns=["Timestamp"]).describe()
    return df.describe()
