#!/usr/bin/env python3
"""Performs K-means clustering on a dataset"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset

    Parameters:
    - X: np.ndarray of shape (n, d)
    - k: number of clusters
    - iterations: max number of iterations

    Returns:
    - C: np.ndarray of shape (k, d), centroids
    - clss: np.ndarray of shape (n,), index of cluster for each data point
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
        not isinstance(k, int) or k <= 0 or
        not isinstance(iterations, int) or iterations <= 0):
        return None, None

    n, d = X.shape
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    C = np.random.uniform(min_vals, max_vals, (k, d))

    for _ in range(iterations):
        # Compute distances and assign points
        dists = np.linalg.norm(X[:, None] - C[None, :], axis=2)
        clss = np.argmin(dists, axis=1)

        C_prev = C.copy()

        # Vectorized centroid update
        mask = (clss[:, None] == np.arange(k)).astype(int)
        counts = mask.sum(axis=0)

        # Avoid division by zero
        counts[counts == 0] = 1

        new_C = (mask.T @ X) / counts[:, None]

        # Handle empty clusters by reinitializing
        empty = (mask.sum(axis=0) == 0)
        if np.any(empty):
            new_C[empty] = np.random.uniform(min_vals, max_vals, (empty.sum(), d))

        if np.allclose(C, new_C):
            break

        C = new_C

    return C, clss
