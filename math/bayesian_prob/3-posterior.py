#!/usr/bin/env python3
"""
This module calculates the posterior probability of each
hypothetical probability using Bayes' theorem.
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

    comb = np.math.factorial(n) / (np.math.factorial(x) * np.math.factorial(n - x))
    return comb * (P ** x) * ((1 - P) ** (n - x))


def marginal(x, n, P, Pr):
    """
    Calculates the marginal probability of observing the data.
    """
    L = likelihood(x, n, P)
    return np.sum(L * Pr)


def posterior(x, n, P, Pr):
    """
    Calculates the posterior probability of each hypothesis in P
    given x and n using Bayes' Rule.

    Returns:
        numpy.ndarray: posterior probabilities

    Raises:
        TypeError or ValueError as specified in the task
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

    L = likelihood(x, n, P)
    M = marginal(x, n, P, Pr)

    return (L * Pr) / M
