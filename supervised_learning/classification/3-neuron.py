#!/usr/bin/env python3

"""
This module defines a class Neuron for binary classification
using logistic regression.
The class is initialized with the number of input features
and computes the forward
propagation, cost, and other relevant metrics for
training a single neuron.
"""

import numpy as np


class Neuron:
    """
    A class representing a single neuron for
    binary classification.
    The neuron uses logistic regression for
    the forward propagation and cost calculation.
    """

    def __init__(self, nx):
        """
        Initializes the Neuron instance with random weights and a bias.

        Args:
            nx (int): The number of input features to the neuron.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Initialize weights, bias, and activation output
        self.__W = np.random.randn(1, nx)  # Weights initialized randomly
        self.__b = 0  # Bias initialized to 0
        self.__A = 0  # Activation output initialized to 0

    @property
    def W(self):
        """
        Getter for the weights.

        Returns:
            numpy.ndarray: The weights of the neuron.
        """
        return self.__W

    @property
    def b(self):
        """
        Getter for the bias.

        Returns:
            float: The bias of the neuron.
        """
        return self.__b

    @property
    def A(self):
        """
        Getter for the activation output.

        Returns:
            numpy.ndarray: The activation output of the neuron.
        """
        return self.__A

    @A.setter
    def A(self, value):
        """
        Setter for the activation output,
        which is not allowed.

        Args:
            value: The value to set for the activation output
            (which is disallowed).

        Raises:
            AttributeError: Always raises an error
            because activation output should not be set manually.
        """
        raise AttributeError("can't set attribute")

    def forward_prop(self, X):
        """
        Calculate the forward propagation of the neuron.

        Args:
            X (numpy.ndarray): The input data, shape (nx, m),
            where nx is the number of input features
                               and m is the number of examples.

        Returns:
            numpy.ndarray: The activated output of the neuron
            after applying the sigmoid activation function.
        """
        Z = np.dot(self.__W, X) + self.__b  # Linear transformation
        self.__A = 1 / (1 + np.exp(-Z))  # Sigmoid activation function
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using
        logistic regression.

        Args:
            Y (numpy.ndarray): True labels for the input data,
            shape (1, m).
            A (numpy.ndarray): Activated output of the neuron,
            shape (1, m).

        Returns:
            float: The cost of the model using the
            logistic regression cost function.
        """
        m = Y.shape[1]  # Number of examples
        # Compute the cost using the logistic regression formula
        cost = - (1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost
