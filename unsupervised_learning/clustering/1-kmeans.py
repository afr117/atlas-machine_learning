#!/usr/bin/env python3
"""Performs K-means clustering on a dataset."""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    return np.random.uniform(low=min_vals, high=max_vals, size=(k, X.shape[1])), min_vals, max_vals


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    Args:
        X (np.ndarray): shape (n, d), dataset
        k (int): number of clusters
        iterations (int): max number of iterations

    Returns:
        C (np.ndarray): (k, d), cluster centroids
        clss (np.ndarray): (n,), index of the cluster each point belongs to
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
        not isinstance(k, int) or k <= 0 or
        not isinstance(iterations, int) or iterations <= 0):
        return None, None

    n, d = X.shape
    C, min_vals, max_vals = initialize(X, k)
    if C is None:
        return None, None

    for _ in range(iterations):
        # Assign each point to the closest centroid
        dist = np.linalg.norm(X[:, None] - C[None, :], axis=2)
        clss = np.argmin(dist, axis=1)

        # Store previous centroids
        C_prev = C.copy()

        # Update step
        for i in range(k):
            if np.any(clss == i):
                C[i] = np.mean(X[clss == i], axis=0)
            else:
                # Reinitialize empty cluster using same bounds
                C[i] = np.random.uniform(min_vals, max_vals)

        # Check for convergence
        if np.allclose(C, C_prev):
            break

    return C, clss
