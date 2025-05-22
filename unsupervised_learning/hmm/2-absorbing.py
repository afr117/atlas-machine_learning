#!/usr/bin/env python3
"""
Determines if a Markov chain is absorbing
"""
import numpy as np


def absorbing(P):
    """
    Determines if a Markov chain is absorbing

    Parameters:
    - P: numpy.ndarray of shape (n, n) representing the transition matrix

    Returns:
    - True if the chain is absorbing
    - False otherwise or on failure
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return False

    n, m = P.shape
    if n != m:
        return False

    if not np.allclose(P.sum(axis=1), 1):
        return False

    absorbing_states = (
        np.isclose(np.diag(P), 1) &
        np.all(np.isclose(P - np.eye(n), 0), axis=1)
    )

    if not np.any(absorbing_states):
        return False

    reach = np.copy(P)
    for _ in range(n):
        reach = np.matmul(reach, P)

    can_reach_absorbing = np.any(reach[:, absorbing_states], axis=1)
    return np.all(can_reach_absorbing)
