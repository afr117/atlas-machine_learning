#!/usr/bin/env python3
"""Performs the expectation step in the EM algorithm for GMM"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Performs the expectation step in the EM algorithm for a GMM

    Parameters:
    - X: np.ndarray of shape (n, d), data set
    - pi: np.ndarray of shape (k,), priors for each cluster
    - m: np.ndarray of shape (k, d), centroid means for each cluster
    - S: np.ndarray of shape (k, d, d), covariance matrices for each cluster

    Returns:
    - g: np.ndarray of shape (k, n), posterior probabilities (responsibilities)
    - l: total log likelihood
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None
    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None

    try:
        k = pi.shape[0]
        n = X.shape[0]
        g = np.zeros((k, n))
        for i in range(k):
            g[i] = pi[i] * pdf(X, m[i], S[i])

        total = np.sum(g, axis=0)
        l = np.sum(np.log(total))
        g /= total

        return g, l
    except Exception:
        return None, None
