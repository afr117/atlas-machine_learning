#!/usr/bin/env python3
"""Performs the maximization step in the EM algorithm for a GMM"""
import numpy as np


def maximization(X, g):
    """
    Performs the maximization step in the EM algorithm for a GMM

    Parameters:
    - X: np.ndarray of shape (n, d), dataset
    - g: np.ndarray of shape (k, n), posterior probabilities

    Returns:
    - pi: np.ndarray of shape (k,), updated priors
    - m: np.ndarray of shape (k, d), updated means
    - S: np.ndarray of shape (k, d, d), updated covariances
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None
    n, d = X.shape
    k, n_check = g.shape
    if n != n_check:
        return None, None, None

    # Sum of responsibilities for each cluster
    Nk = np.sum(g, axis=1)

    # Updated priors
    pi = Nk / n

    # Updated means
    m = (g @ X) / Nk[:, np.newaxis]

    # Updated covariances
    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        weighted_diff = g[i][:, np.newaxis] * diff
        S[i] = (weighted_diff.T @ diff) / Nk[i]

    return pi, m, S
