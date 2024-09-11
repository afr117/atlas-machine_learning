#!/usr/bin/env python3
"""
Normal Distribution class
"""


class Normal:
    """Represents a Normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize the Normal distribution.
        Args:
            data: List of data to estimate the distribution.
            mean: The mean of the distribution.
            stddev: The standard deviation of the distribution.
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = variance ** 0.5

    def z_score(self, x):
        """Calculates the z-score of a given x-value."""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score."""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value."""
        pi = 3.1415926536
        e = 2.7182818285
        numerator = e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2)
        denominator = self.stddev * ((2 * pi) ** 0.5)
        return numerator / denominator

    def erf(self, z):
        """
        Approximate the error function for a given z.
        """
        pi = 3.1415926536
        term1 = z - (z ** 3) / 3
        term2 = (z ** 5) / 10 - (z ** 7) / 42
        term3 = (z ** 9) / 216
        term = (2 / (pi ** 0.5)) * (term1 + term2 + term3)
        return term

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value.
        Args:
            x: The x-value
        Returns:
            The CDF value for x
        """
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))
        return 0.5 * (1 + self.erf(z))
