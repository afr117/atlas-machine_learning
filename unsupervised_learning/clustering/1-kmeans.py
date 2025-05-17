#!/usr/bin/env python3
"""Performs K-means clustering on a dataset"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset

    Parameters:
    - X: np.ndarray of shape (n, d), dataset
    - k: int, number of clusters
    - iterations: int, maximum number of iterations

    Returns:
    - C: np.ndarray of shape (k, d), centroid means
    - clss: np.ndarray of shape (n,), index of cluster each data point belongs to
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
        not isinstance(k, int) or k <= 0 or
        not isinstance(iterations, int) or iterations <= 0):
        return None, None

    n, d = X.shape

    # Initialize centroids with a uniform distribution between min and max of X
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    C = np.random.uniform(min_vals, max_vals, (k, d))

    for i in range(iterations):
        # Compute distances and assign clusters
        dists = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(dists, axis=1)

        # Store previous centroids for convergence check
        C_prev = C.copy()

        for j in range(k):
            points = X[clss == j]
            if points.shape[0] > 0:
                C[j] = np.mean(points, axis=0)
            else:
                # Reinitialize empty cluster centroid
                C[j] = np.random.uniform(min_vals, max_vals)

        # Check for convergence
        if np.allclose(C, C_prev):
            break

    return C, clss
