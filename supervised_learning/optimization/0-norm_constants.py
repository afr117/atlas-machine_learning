#!/usr/bin/env python3
"""
Calculates the normalization constants (mean and standard deviation)
of a given dataset matrix.
"""
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization constants of a matrix.

    Parameters:
    X (numpy.ndarray): A matrix of shape
    (m, nx) where m is the number of data points
                       and nx is the number of features.

    Returns:
    tuple: The mean and standard deviation of each feature, respectively.
    """
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    return mean, std_dev
