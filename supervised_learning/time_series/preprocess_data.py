#!/usr/bin/env python3
"""
Preprocess raw BTC datasets from Coinbase and Bitstamp.

- Removes unneeded columns (open, high, low, volume data)
- Keeps only the closing price for prediction
- Merges datasets to fill missing values
- Sorts by timestamp and removes duplicates
- Standardizes features using z-score normalization
- Saves numpy arrays for model training
"""

import numpy as np
import pandas as pd


def load_and_clean(path):
    """
    Loads and cleans a raw BTC dataset.

    Args:
        path (str): CSV file path.

    Returns:
        pandas.DataFrame: Cleaned dataset.
    """
    df = pd.read_csv(path)
    df = df.dropna()
    df = df.rename(columns={'Timestamp': 'time'})
    df = df.sort_values('time')

    # Keep only timestamp + close price (best predictor)
    df = df[['time', 'Close']]
    df = df.drop_duplicates(subset='time')

    return df


def standardize(series):
    """
    Standardizes a time series using z-score normalization.

    Args:
        series (numpy.ndarray): 1D array of values.

    Returns:
        tuple: (standardized array, mean, std)
    """
    mu = series.mean()
    sigma = series.std()
    return (series - mu) / sigma, mu, sigma


def main():
    """Main preprocessing pipeline."""
    coinbase = load_and_clean("coinbase.csv")
    bitstamp = load_and_clean("bitstamp.csv")

    # Merge using time index; fill gaps w/ available exchange data
    merged = pd.merge_asof(
        coinbase,
        bitstamp,
        on='time',
        direction='nearest',
        suffixes=('_cb', '_bs')
    )

    # Average both exchange closing prices
    merged['close'] = merged[['Close_cb', 'Close_bs']].mean(axis=1)
    merged = merged[['time', 'close']]

    # Standardize closing price
    std_close, mu, sigma = standardize(merged['close'].values)

    # Save preprocessed arrays
    np.save("close_standardized.npy", std_close)
    np.save("close_mu.npy", np.array([mu]))
    np.save("close_sigma.npy", np.array([sigma]))

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
