#!/usr/bin/env python3
"""
This module provides a function to decode a one-hot matrix into a vector of labels.
"""

import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot encoded matrix into a vector of labels.

    Parameters:
    - one_hot (numpy.ndarray): A 2D array of shape (classes, m), where:
        - `classes` is the number of classes.
        - `m` is the number of examples.

    Returns:
    - numpy.ndarray: A 1D array of shape (m,) containing the decoded class labels.
    - None: If the input is invalid (not a proper one-hot matrix).

    The function ensures:
    - The input is a 2D numpy array.
    - All elements are binary (0 or 1).
    - Each column contains exactly one `1` (valid one-hot encoding).
    """
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None
    if not np.all((one_hot == 0) | (one_hot == 1)):
        # Ensure binary values (0 or 1)
        return None
    if not np.all(np.sum(one_hot, axis=0) == 1):
        # Ensure valid one-hot encoding (only one '1' per column)
        return None

    return np.argmax(one_hot, axis=0)
