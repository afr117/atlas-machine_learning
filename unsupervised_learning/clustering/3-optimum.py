#!/usr/bin/env python3
"""Finds the optimum number of clusters by variance"""
import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance

    Parameters:
    - X: np.ndarray of shape (n, d), data set
    - kmin: int, minimum number of clusters (inclusive)
    - kmax: int, maximum number of clusters (inclusive)
    - iterations: int, max number of iterations for K-means

    Returns:
    - results: list of outputs of K-means for each cluster size
    - d_vars: list of variance differences from smallest k
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax <= 0):
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n = X.shape[0]
    if kmax is None:
        kmax = n

    if kmax - kmin < 1:
        return None, None

    results = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        res = kmeans(X, k, iterations)
        if res is None:
            return None, None
        results.append(res)
        d_vars.append(variance(X, res[0]))

    d_vars = [d_vars[0] - v for v in d_vars]
    return results, d_vars
