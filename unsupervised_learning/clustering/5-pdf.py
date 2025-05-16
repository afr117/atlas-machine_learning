#!/usr/bin/env python3
"""Calculates the PDF of a multivariate Gaussian distribution"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a multivariate Gaussian

    Parameters:
    - X: np.ndarray of shape (n, d), data points
    - m: np.ndarray of shape (d,), mean of the distribution
    - S: np.ndarray of shape (d, d), covariance of the distribution

    Returns:
    - P: np.ndarray of shape (n,), PDF values for each data point
    """
    if not isinstance(X, np.ndarray) or not isinstance(m, np.ndarray) \
       or not isinstance(S, np.ndarray):
        return None
    if len(X.shape) != 2 or len(m.shape) != 1 or len(S.shape) != 2:
        return None
    n, d = X.shape
    if m.shape[0] != d or S.shape != (d, d):
        return None

    try:
        det_S = np.linalg.det(S)
        if det_S <= 0:
            return None
        inv_S = np.linalg.inv(S)
        norm_const = 1.0 / np.sqrt(((2 * np.pi) ** d) * det_S)
        diff = X - m
        exp_term = np.sum(diff @ inv_S * diff, axis=1)
        P = norm_const * np.exp(-0.5 * exp_term)
        P = np.maximum(P, 1e-300)
        return P
    except Exception:
        return None
