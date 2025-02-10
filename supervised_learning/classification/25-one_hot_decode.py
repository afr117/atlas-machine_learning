#!/usr/bin/env python3

import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of labels"""
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None
    if not np.all((one_hot == 0) | (one_hot == 1)):
        # Ensure binary values
        return None
    if not np.all(np.sum(one_hot, axis=0) == 1):
        # Ensure valid one-hot encoding
        return None

    return np.argmax(one_hot, axis=0)
