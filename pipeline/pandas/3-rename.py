#!/usr/bin/env python3
"""Module that renames and formats a pandas DataFrame."""

import pandas as pd


def rename(df):
    """
    Rename the Timestamp column to Datetime, convert it to datetime,
    and display only the Datetime and Close columns.

    Args:
        df (pd.DataFrame): DataFrame containing a 'Timestamp' column.

    Returns:
        pd.DataFrame: The modified DataFrame with only Datetime and Close.
    """
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    return df[["Datetime", "Close"]]
