#!/usr/bin/env python3
"""
Defines a single neuron performing binary classification
"""
import numpy as np


class Neuron:
    """
    Class that defines a single neuron for binary classification
    """
    def __init__(self, nx):
        """
        Initializes the neuron
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(1, nx)  # No seed, fully random
        self.b = 0
        self.A = 0
