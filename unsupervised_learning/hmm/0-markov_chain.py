#!/usr/bin/env python3
"""
Determines the probability of a Markov Chain being in a particular
state after a given number of iterations
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a Markov Chain being in a particular
    state after a given number of iterations

    Parameters:
    - P: 2D numpy.ndarray of shape (n, n) representing the transition matrix
    - s: numpy.ndarray of shape (1, n) representing the initial state
    - t: number of iterations (positive int)

    Returns:
    - numpy.ndarray of shape (1, n) representing the probability
      of being in a specific state after t iterations
    - or None on failure
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    n, m = P.shape
    if n != m:
        return None

    if not isinstance(s, np.ndarray) or s.shape != (1, n):
        return None

    if not isinstance(t, int) or t < 1:
        return None

    for _ in range(t):
        s = np.matmul(s, P)

    return s
