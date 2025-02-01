#!/usr/bin/env python3
"""
A module defining a single neuron for binary classification.
"""

import numpy as np


class Neuron:
    """
    A class representing a single neuron for binary classification.

    This class models a single neuron for binary classification,
    including initialization of its weights, bias, and activation output.
    The weights are initialized using a random normal distribution,
    while the bias and activation output are initialized to zero.

    Attributes:
        __W (numpy.ndarray): The weights vector for the neuron.
        __b (float): The bias of the neuron.
        __A (float): The activated output of the neuron.
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
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        Getter for the weights attribute.

        Returns:
            numpy.ndarray: The weights vector of the neuron.
        """
        return self.__W

    @property
    def b(self):
        """
        Getter for the bias attribute.

        Returns:
            float: The bias of the neuron.
        """
        return self.__b

    @property
    def A(self):
        """
        Getter for the activation output attribute.

        Returns:
            float: The activated output of the neuron.
        """
        return self.__A

    @A.setter
    def A(self, value):
        """
        Prevent setting the activation output attribute.

        Args:
            value: The value to set (which is ignored).

        Raises:
            AttributeError: Always raises since setting is not allowed.
        """
        raise AttributeError("can't set attribute")

    def forward_prop(self, X):
        """
        Calculate the forward propagation of the neuron.

        Args:
            X (numpy.ndarray): The input data, with shape (nx, m), where
                               nx is the number of input features and
                               m is the number of examples.

        Returns:
            numpy.ndarray: The activated output of the neuron.
        """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))  # Sigmoid activation function
        return self.__A
