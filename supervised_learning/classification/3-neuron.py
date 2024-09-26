#!/usr/bin/env python3
"""
Neuron class for binary classification with cost computation.
"""

import numpy as np

class Neuron:
    """
    Defines a single neuron performing binary classification.
    """
    def __init__(self, nx):
        """
        Initialize the neuron with the number of input features.

        Parameters:
        nx (int): Number of input features to the neuron.

        Raises:
        ValueError: If nx is not a positive integer.
        """
        if not isinstance(nx, int) or nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)  # Initialize weights
        self.__b = 0  # Initialize bias
        self.__A = 0  # Initialize activated output

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.

        Parameters:
        X (numpy.ndarray): Input data with shape (nx, m), where nx is the number of input features,
                           and m is the number of examples.

        Returns:
        numpy.ndarray: Activated output of the neuron with shape (1, m).
        """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))  # Sigmoid activation function
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Parameters:
        Y (numpy.ndarray): Correct labels with shape (1, m).
        A (numpy.ndarray): Activated output with shape (1, m).

        Returns:
        float: The cost of the model.
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A + 1.0000001) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    @property
    def A(self):
        """
        Getter for the private attribute __A.
        """
        return self.__A
