#!/usr/bin/env python3
"""
Exponential Distribution class
"""


class Exponential:
    """Represents an Exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize the Exponential distribution.
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
            self.lambtha = 1 / (sum(data) / len(data))  # Calculate lambtha from data

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period x
        Args:
            x: Time period (must be non-negative)
        Returns:
            The PDF value for x
        """
        if x < 0:
            return 0

        e = 2.7182818285  # Euler's number
        return self.lambtha * (e ** (-self.lambtha * x))

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period x
        Args:
            x: Time period (must be non-negative)
        Returns:
            The CDF value for x
        """
        if x < 0:
            return 0

        e = 2.7182818285  # Euler's number
        return 1 - (e ** (-self.lambtha * x))
