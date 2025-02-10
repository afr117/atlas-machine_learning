#!/usr/bin/env python3
"""
This module provides a function for one-hot encoding a numeric label vector.
"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    Parameters:
    - Y (numpy.ndarray): A 1D array of shape (m,) containing numeric class labels.
    - classes (int): The total number of classes.

    Returns:
    - numpy.ndarray: A one-hot encoded matrix of shape (classes, m) where:
        - Each column represents a single label as a one-hot vector.
    - None: If the input is invalid (e.g., incorrect type, dimension, or class range).
    """
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        return None
    if Y.ndim != 1 or classes < np.max(Y) + 1:
        return None

    one_hot = np.zeros((classes, Y.shape[0]))
    one_hot[Y, np.arange(Y.shape[0])] = 1
    return one_hot
