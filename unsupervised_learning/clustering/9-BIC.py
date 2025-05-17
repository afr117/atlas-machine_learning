#!/usr/bin/env python3
"""Finds the best number of clusters using BIC for a GMM"""
import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the BIC

    Returns:
    - best_k: best value for k based on BIC
    - best_result: tuple of (pi, m, S) for best model
    - l: array of log likelihoods
    - b: array of BICs
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax < kmin):
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    if kmax is None:
        kmax = n

    l = []
    b = []
    results = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        if pi is None or m is None or S is None or g is None or log_likelihood is None:
            return None, None, None, None

        if verbose:
            print(f"Log Likelihood after {len(l) + 1 + 10} iterations: {log_likelihood:.5f}")

        l.append(log_likelihood)
        results.append((pi, m, S))

        # Number of parameters
        p = (k - 1) + (k * d) + (k * d * (d + 1) / 2)
        bic = p * np.log(n) - 2 * log_likelihood
        b.append(bic)

    l = np.array(l)
    b = np.array(b)
    best_index = np.argmin(b)
    best_k = best_index + kmin
    best_result = results[best_index]

    return best_k, best_result, l, b
