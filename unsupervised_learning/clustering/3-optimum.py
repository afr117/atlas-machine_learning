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
        X (np.ndarray): Dataset of shape (n, d)
        kmin (int): Minimum number of clusters (inclusive)
        kmax (int): Maximum number of clusters (inclusive)
        iterations (int): Max number of iterations for K-means

    Returns:
        results (list): Outputs of K-means for each k
        d_vars (list): Difference in variance from smallest k
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
        not isinstance(kmin, int) or kmin < 1 or
        (kmax is not None and (not isinstance(kmax, int) or kmax < kmin)) or
        not isinstance(iterations, int) or iterations < 1):
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    results = []
    d_vars = []
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
