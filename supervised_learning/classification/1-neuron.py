#!/usr/bin/env python3

"""
Neuron Module
This module contains the Neuron class that models a single neuron
for binary classification, handling initialization of weights,
bias, and activation output.
"""

import numpy as np


class Neuron:
    """
    A class representing a single neuron for binary classification.

    This class models a single neuron for binary classification,
    including the initialization of its weights, bias, and activation output.
    The weights are initialized using a random normal distribution,
    while the bias and activation output are initialized to zero.

    Attributes:
        __W (numpy.ndarray): The weights vector for the neuron.
        __b (float): The bias of the neuron.
        __A (float): The activated output (prediction) of the neuron.
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
        # The weights vector is initialized
        # with random values from a standard normal distribution
        self.__b = 0
        # The bias is initialized to zero
        self.__A = 0
        # The activation output is initialized to zero

    @property
    def W(self):
        """
        Getter for the weights vector attribute.

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
            float: The activated output (prediction) of the neuron.
        """
        return self.__A

    @A.setter
    def A(self, value):
        """
        Prevent setting the activation output attribute.

        This method raises an AttributeError if an attempt is made to modify
        the activation output directly since it's intended to be read-only.

        Args:
            value: The value to set (which is ignored).

        Raises:
            AttributeError: Always raises since setting is not allowed.
        """
        raise AttributeError("can't set attribute")
