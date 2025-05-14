#!/usr/bin/env python3
"""
Calculates the total intra-cluster variance for a dataset.
"""

import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance.

    Args:
        X (np.ndarray): shape (n, d), data points
        C (np.ndarray): shape (k, d), cluster centroids

    Returns:
        float: total variance, or None on failure
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
        not isinstance(C, np.ndarray) or C.ndim != 2 or
        X.shape[1] != C.shape[1]):
        return None

    # Compute distance from each point to each centroid
    dist = np.linalg.norm(X[:, None] - C[None, :], axis=2)

    # Assign points to the closest centroid
    clss = np.argmin(dist, axis=1)

    # Compute squared distances to the assigned centroids
    sq_dists = np.sum((X - C[clss]) ** 2)

    return sq_dists
