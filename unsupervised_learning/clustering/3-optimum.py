#!/usr/bin/env python3
"""
Finds the optimal number of clusters for a dataset using K-means.
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Finds the optimal number of clusters based on variance.

    Args:
        X (np.ndarray): shape (n, d), dataset
        kmin (int): minimum number of clusters to check
        kmax (int): maximum number of clusters to check
        iterations (int): max iterations for K-means

    Returns:
        results (list): (C, clss) from K-means for each k
        d_vars (list): variance differences from kmin
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None
    if kmax is not None:
        if not isinstance(kmax, int) or kmax < kmin:
            return None, None
    else:
        kmax = X.shape[0]

    if not isinstance(iterations, int) or iterations < 1:
        return None, None

    results = []
    variances = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None
        results.append((C, clss))
        variances.append(variance(X, C))

    base_var = variances[0]
    d_vars = [v - base_var for v in variances]

    return results, d_vars
