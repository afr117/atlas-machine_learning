#!/usr/bin/env python3
"""Performs the maximization step in the EM algorithm for a GMM"""
import numpy as np


def maximization(X, g):
    """
    Performs the maximization step in the EM algorithm for a GMM

    Parameters:
    - X: np.ndarray of shape (n, d), data set
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

    Nk = np.sum(g, axis=1)
    if np.any(Nk == 0):
        return None, None, None

    pi = Nk / n
    m = (g @ X) / Nk[:, None]

    # Vectorized covariance computation with einsum
    X_exp = X[None, :, :]         # (1, n, d)
    m_exp = m[:, None, :]         # (k, 1, d)
    diff = X_exp - m_exp          # (k, n, d)
    weighted_diff = diff * g[:, :, None]  # (k, n, d)
    S = np.einsum('kni,knj->kij', weighted_diff, diff) / Nk[:, None, None]

    return pi, m, S
