#!/usr/bin/env python3
"""
This module calculates the marginal probability of observing
x outcomes in n trials under a set of prior probabilities.
"""

import numpy as np


def likelihood(x, n, P):
    """
    Calculates likelihood of observing x given n and probabilities P.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Binomial coefficient
    comb = np.math.factorial(n) / (np.math.factorial(x) * np.math.factorial(n - x))
    return comb * (P ** x) * ((1 - P) ** (n - x))


def marginal(x, n, P, Pr):
    """
    Calculates the marginal probability of observing x outcomes in n trials.

    Args:
        x (int): number of patients with side effects
        n (int): total number of patients observed
        P (np.ndarray): 1D array of hypothetical probabilities
        Pr (np.ndarray): 1D array of prior beliefs

    Returns:
        float: the marginal probability

    Raises:
        TypeError or ValueError: as specified by the task
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    inter = likelihood(x, n, P) * Pr
    return np.sum(inter)
