#!/usr/bin/env python3
"""
This module defines a function that calculates the mean and
covariance matrix of a given dataset.
"""

import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a dataset.

    Args:
        X (np.ndarray): shape (n, d), where
            n is the number of data points
            d is the number of dimensions per data point

    Returns:
        mean (np.ndarray): shape (1, d) - mean vector of the dataset
        cov (np.ndarray): shape (d, d) - covariance matrix of the dataset

    Raises:
        TypeError: if X is not a 2D numpy.ndarray
        ValueError: if X has fewer than 2 data points
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - mean
    cov = np.matmul(X_centered.T, X_centered) / (n - 1)

    return mean, cov
