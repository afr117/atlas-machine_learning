#!/usr/bin/env python3

"""
This module defines a class Neuron that models a single neuron
for binary classification.
The class is initialized with a number of input features
and initializes its weights,
bias, and activation output.
"""

import numpy as np


class Neuron:
    """
    A class that defines a single neuron for binary classification.
    """

    def __init__(self, nx):
        """
        Initializes the Neuron instance.

        Args:
            nx (int): The number of input features.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        # Ensure nx is an integer
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        # Ensure nx is a positive integer
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Initialize weights W with a random normal distribution
        self.W = np.random.randn(1, nx)

        # Initialize bias b to 0
        self.b = 0

        # Initialize activation output A to 0
        self.A = 0
