#!/usr/bin/env python3

"""Neuron class performing binary classification"""

import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """
        Initializes the Neuron

        Parameters:
        nx (int): The number of input features to the neuron

        Raises:
        TypeError: If nx is not an integer
        ValueError: If nx is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Initialize weights, bias, and activated output
        self.W = np.random.randn(1, nx)  # Weights initialized with random normal distribution
        self.b = 0  # Bias initialized to 0
        self.A = 0  # Activated output initialized to 0
