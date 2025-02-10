#!/usr/bin/env python3
"""
This module defines a NeuralNetwork class that implements a
binary classification model with one hidden layer.
"""

import numpy as np


class NeuralNetwork:
    """
    Defines a neural network with
    one hidden layer performing binary classification.

    Attributes:
    - W1 (numpy.ndarray): Weights for the hidden layer.
    - b1 (numpy.ndarray): Bias for the hidden layer.
    - A1 (float): Activated output of the hidden layer.
    - W2 (numpy.ndarray): Weights for the output layer.
    - b2 (float): Bias for the output layer.
    - A2 (float): Activated output of the output layer.
    """

    def __init__(self, nx, nodes):
        """
        Initializes a neural network.

        Parameters:
        - nx (int): Number of input features.
        - nodes (int): Number of nodes in the hidden layer.

        Raises:
        - TypeError: If nx is not an integer.
        - ValueError: If nx is less than 1.
        - TypeError: If nodes is not an integer.
        - ValueError: If nodes is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter method for W1 (hidden layer weights)."""
        return self.__W1

    @property
    def b1(self):
        """Getter method for b1 (hidden layer bias)."""
        return self.__b1

    @property
    def A1(self):
        """Getter method for A1 (hidden layer activation output)."""
        return self.__A1

    @property
    def W2(self):
        """Getter method for W2 (output layer weights)."""
        return self.__W2

    @property
    def b2(self):
        """Getter method for b2 (output layer bias)."""
        return self.__b2

    @property
    def A2(self):
        """Getter method for A2 (output layer activation output)."""
        return self.__A2

    def forward_prop(self, X):
        """
        Performs forward propagation.

        Parameters:
        - X (numpy.ndarray): Input data of shape (nx, m).

        Returns:
        - A1 (numpy.ndarray): Activated output of the hidden layer.
        - A2 (numpy.ndarray): Activated output of the output layer.
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Computes the cost using logistic regression.

        Parameters:
        - Y (numpy.ndarray): True labels of shape (1, m).
        - A (numpy.ndarray): Activated output of the output layer.

        Returns:
        - cost (float): Cost function value.
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost
