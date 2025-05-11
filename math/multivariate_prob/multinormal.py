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
            self.mean (np.ndarray): shape (d, 1) - mean vector
            self.cov (np.ndarray): shape (d, d) - covariance matrix

        Raises:
            TypeError: if data is not a 2D numpy.ndarray
            ValueError: if data has fewer than 2 points
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Mean vector: shape (d, 1)
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Center data
        data_centered = data - self.mean

        # Covariance matrix: shape (d, d)
        self.cov = np.matmul(data_centered, data_centered.T) / (n - 1)
