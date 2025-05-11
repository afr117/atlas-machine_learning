#!/usr/bin/env python3
"""
This module defines a function to calculate the correlation matrix
from a given covariance matrix.
"""

import numpy as np


def correlation(C):
    """
    Calculates the correlation matrix from a covariance matrix.

    Args:
        C (np.ndarray): shape (d, d), the covariance matrix

    Returns:
        np.ndarray: shape (d, d), the correlation matrix

    Raises:
        TypeError: if C is not a numpy.ndarray
        ValueError: if C is not a 2D square matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    std_dev = np.sqrt(np.diag(C))
    denom = np.outer(std_dev, std_dev)

    return C / denom
