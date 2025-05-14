#!/usr/bin/env python3
"""
This module performs K-means clustering on a dataset.
"""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    return np.random.uniform(low=min_vals, high=max_vals, size=(k, d))


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    Args:
        X (np.ndarray): shape (n, d), dataset
        k (int): number of clusters
        iterations (int): max number of iterations

    Returns:
        C (np.ndarray): shape (k, d), centroids
        clss (np.ndarray): shape (n,), index of cluster each point belongs to
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
        not isinstance(k, int) or k <= 0 or
        not isinstance(iterations, int) or iterations <= 0):
        return None, None

    n, d = X.shape
    C = initialize(X, k)
    if C is None:
        return None, None

    for _ in range(iterations):
        # Assign points to closest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        # Save current centroids to check convergence
        C_prev = C.copy()

        # Update centroids
        for i in range(k):
            points = X[clss == i]
            if points.size == 0:
                C[i] = np.random.uniform(np.min(X, axis=0),
                                         np.max(X, axis=0))
            else:
                C[i] = np.mean(points, axis=0)

        # Check for convergence
        if np.allclose(C, C_prev):
            break

    return C, clss
