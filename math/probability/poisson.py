#!/usr/bin/env python3
"""
Poisson Distribution class
"""


class Poisson:
    """Represents a Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize the Poisson distribution.
        Args:
            data: List of data to estimate the distribution.
            lambtha: The expected number of occurrences in a given timeframe.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of 'successes'
        Args:
            k: number of occurrences (successes)
        Returns:
            The PMF value for k
        """
        if k < 0:
            return 0
        k = int(k)
        e = 2.7182818285
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        pmf_value = (self.lambtha ** k) * (e ** -self.lambtha) / factorial
        return pmf_value
