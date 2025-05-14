#!/usr/bin/env python3
"""
Finds the optimal number of clusters using the variance difference method.
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Finds the optimal number of clusters by variance.

    Args:
        X (np.ndarray): shape (n, d), dataset
        kmin (int): min number of clusters to check (inclusive)
        kmax (int): max number of clusters to check (inclusive)
        iterations (int): max iterations for K-means

    Returns:
        results (list): outputs of K-means for each k
        d_vars (list): delta variance from smallest k
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
        not isinstance(kmin, int) or kmin < 1 or
        (kmax is not None and
         (not isinstance(kmax, int) or kmax < kmin)) or
        not isinstance(iterations, int) or iterations < 1):
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    results = []
    d_vars = []
    base_var = None

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None
        var = variance(X, C)
        if base_var is None:
            base_var = var
        results.append((C, clss))
        d_vars.append(var - base_var)

    return results, d_vars
