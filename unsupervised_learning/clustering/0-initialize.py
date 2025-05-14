#!/usr/bin/env python3
"""
This module initializes cluster centroids for K-means clustering
using a multivariate uniform distribution.
"""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means.

    Args:
        X (np.ndarray): shape (n, d) containing the dataset
        k (int): number of clusters

    Returns:
        np.ndarray of shape (k, d) containing initialized centroids,
        or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    return np.random.uniform(low=min_vals, high=max_vals, size=(k, d))
