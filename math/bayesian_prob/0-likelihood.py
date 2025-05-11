#!/usr/bin/env python3
"""
This module calculates the likelihood of observing x successes
in n trials for a range of hypothetical probabilities.
"""

import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of observing x outcomes from n trials
    given a 1D array of probabilities P using the binomial distribution.

    Args:
        x (int): number of observed successes
        n (int): total number of trials
        P (np.ndarray): 1D array of hypothetical probabilities

    Returns:
        np.ndarray: array of likelihood values for each probability in P

    Raises:
        TypeError: if P is not a 1D numpy.ndarray
        ValueError: if x or n is invalid or values in P are not in [0, 1]
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

    # Compute binomial coefficient C(n, x)
    comb = np.math.factorial(n) / (np.math.factorial(x) * np.math.factorial(n - x))

    # Compute likelihoods
    return comb * (P ** x) * ((1 - P) ** (n - x))
