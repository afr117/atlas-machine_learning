#!/usr/bin/env python3
"""
This module defines the MultiNormal class that represents a
Multivariate Normal distribution.
"""

import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution.
    """

    def __init__(self, data):
        """
        Initializes the distribution with data.

        Args:
            data (np.ndarray): shape (d, n) with:
                d: number of dimensions
                n: number of data points

        Sets:
            self.mean (np.ndarray): shape (d, 1)
            self.cov (np.ndarray): shape (d, d)
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        data_centered = data - self.mean
        self.cov = np.matmul(data_centered, data_centered.T) / (n - 1)
        self.d = d

    def pdf(self, x):
        """
        Calculates the PDF at a given data point x.

        Args:
            x (np.ndarray): shape (d, 1) point to evaluate PDF at

        Returns:
            float: value of the PDF

        Raises:
            TypeError: if x is not a numpy.ndarray
            ValueError: if shape of x is not (d, 1)
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if x.shape != (self.d, 1):
            raise ValueError(f"x must have the shape ({self.d}, 1)")

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        diff = x - self.mean

        exp_term = -0.5 * np.matmul(diff.T, np.matmul(inv, diff))
        denom = np.sqrt(((2 * np.pi) ** self.d) * det)

        return float(np.exp(exp_term) / denom)
