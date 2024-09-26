#!/usr/bin/env python3
"""
This module defines a Neuron class for binary classification.

The Neuron class models a single neuron with its weights, bias, and activation output. 
It includes methods to initialize the neuron and manage its parameters.

Attributes:
    Neuron: A class representing a single neuron for binary classification.
"""

import numpy as np


class Neuron:
    """
    A class representing a single neuron for binary classification.

    Attributes:
        W (numpy.ndarray): The weights of the neuron.
        b (float): The bias of the neuron.
        A (float): The activation output of the neuron.
    """

    def __init__(self, nx):
        """
        Initialize a Neuron instance.

        Args:
            nx (int): The number of input features to the neuron.

        Raises:
            TypeError: If `nx` is not an integer.
            ValueError: If `nx` is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Initialize weights, bias, and activation output
        self.W = np.random.randn(1, nx)  # Weights initialized random values
        self.b = 0                       # Bias initialized to zero
        self.A = 0                       # Activation initialized to zero
