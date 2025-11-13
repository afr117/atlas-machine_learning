#!/usr/bin/env python3
"""
Train an RNN to forecast BTC closing price 1 hour ahead using
the previous 24 hours (1440 minutes) of data.

Uses:
- Sliding window dataset creation
- tf.data input pipeline
- LSTM model with MSE loss
"""

import numpy as np
import tensorflow as tf


def make_windows(series, window=1440, horizon=60):
    """
    Create sliding windows for forecasting.

    Args:
        series (np.ndarray): Standardized closing prices.
        window (int): Past time steps to use.
        horizon (int): Future steps to predict.

    Returns:
        tuple: (X, y)
    """
    X, y = [], []
    for i in range(len(series) - window - horizon):
        X.append(series[i:i + window])
        y.append(series[i + window + horizon])
    return np.array(X), np.array(y)


def create_dataset(X, y, batch=64):
    """
    Create a TensorFlow data pipeline.

    Args:
        X (np.ndarray): Input sequences.
        y (np.ndarray): Labels.
        batch (int): Batch size.

    Returns:
        tf.data.Dataset
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(1024).batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(window):
    """
    Build an LSTM-based forecasting model.

    Args:
        window (int): Input sequence length.

    Returns:
        tf.keras.Model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, 1)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model


def main():
    """Main training routine."""
    series = np.load("close_standardized.npy")

    window = 1440   # 24 hours
    horizon = 60    # predict 1 hour ahead

    X, y = make_windows(series, window, horizon)

    X = X[..., np.newaxis]  # (batch, window, 1)
    ds = create_dataset(X, y)

    model = build_model(window)

    model.fit(ds, epochs=5)
    model.save("btc_forecast.h5")
    print("Model training complete.")


if __name__ == "__main__":
    main()
