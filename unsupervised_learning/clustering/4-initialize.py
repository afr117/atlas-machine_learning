#!/usr/bin/env python3
"""Initialize variables for a Gaussian Mixture Model"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model

    Parameters:
    - X: np.ndarray of shape (n, d), data set
    - k: positive int, number of clusters

    Returns:
    - pi: np.ndarray of shape (k,) with priors for each cluster
    - m: np.ndarray of shape (k, d) with centroid means from K-means
    - S: np.ndarray of shape (k, d, d) with identity covariance matrices
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return None, None, None

    _, m = kmeans(X, k)
    d = X.shape[1]
    pi = np.full((k,), 1 / k)
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
