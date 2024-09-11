#!/usr/bin/env python3
"""
Binomial Distribution class
"""


class Binomial:
    """Represents a Binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize the Binomial distribution.
        Args:
            data: List of data to estimate the distribution.
            n: Number of Bernoulli trials.
            p: Probability of success.
        """
        if data is None:
            if not isinstance(n, int) or n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate p as the average of the data
            self.p = sum(data) / len(data)

            # Calculate the variance from the data
            variance = sum([(x - self.p) ** 2 for x in data]) / len(data)

            # Estimate n using variance formula
            self.n = round(self.p * (1 - self.p) / variance)

            # Recalculate p using the estimated n
            self.p = sum(data) / (self.n * len(data))

    def factorial(self, num):
        """Helper function to compute factorial"""
        if num == 0:
            return 1
        factorial = 1
        for i in range(1, num + 1):
            factorial *= i
        return factorial

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes.
        Args:
            k: Number of successes.
        Returns:
            The PMF value for k.
        """
        if k < 0 or k > self.n:
            return 0
        k = int(k)

        # Binomial coefficient: n! / (k!(n-k)!)
        comb = self.factorial(self.n) / (self.factorial(k) * self.factorial(self.n - k))

        # PMF formula
        pmf_value = comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))
        return pmf_value

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of successes.
        Args:
            k: Number of successes.
        Returns:
            The CDF value for k.
        """
        if k < 0:
            return 0
        k = int(k)

        # Sum the PMF values from 0 to k
        cdf_value = sum(self.pmf(i) for i in range(k + 1))
        return cdf_value
