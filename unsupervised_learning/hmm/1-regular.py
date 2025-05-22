#!/usr/bin/env python3
"""
Determines the steady state probabilities of a regular Markov chain
"""
import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular Markov chain

    Parameters:
    - P: numpy.ndarray of shape (n, n) representing the transition matrix

    Returns:
    - numpy.ndarray of shape (1, n) representing steady state probabilities
    - or None on failure
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None

    n, m = P.shape
    if n != m:
        return None

    # Check if P is a valid transition matrix
    if not np.allclose(P.sum(axis=1), 1):
        return None

    # Check if P is regular: some power of P has all positive entries
    power = np.linalg.matrix_power(P, 100)
    if not np.all(power > 0):
        return None

    # Solve for steady state: sP = s → sP - s = 0 → s(P - I) = 0
    # Add constraint: sum(s) = 1
    A = np.vstack((P.T - np.eye(n), np.ones((1, n))))
    b = np.zeros((n + 1,))
    b[-1] = 1

    try:
        steady = np.linalg.lstsq(A, b, rcond=None)[0]
        return steady[np.newaxis, :]
    except Exception:
        return None
