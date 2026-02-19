#!/usr/bin/env python3
"""Module that fills missing values in a DataFrame according to specified rules."""


def fill(df):
    """
    Remove the Weighted_Price column and fill missing values:
      - Close: forward fill
      - High, Low, Open: fill with row's Close
      - Volume_(BTC), Volume_(Currency): fill with 0

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    if "Weighted_Price" in df.columns:
        df = df.drop(columns=["Weighted_Price"])
    df["Close"] = df["Close"].fillna(method="ffill")
    for col in ["High", "Low", "Open"]:
        df[col] = df[col].fillna(df["Close"])
    for col in ["Volume_(BTC)", "Volume_(Currency)"]:
        df[col] = df[col].fillna(0)
    return df
