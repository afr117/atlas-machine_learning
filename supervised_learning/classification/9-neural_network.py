#!/usr/bin/env python3
"""
This module defines a NeuralNetwork class that implements a
binary classification model with one hidden layer.
"""

import numpy as np


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer
    performing binary classification.

    Attributes:
        W1 (numpy.ndarray): Weights for the hidden layer.
        b1 (numpy.ndarray): Bias for the hidden layer.
        A1 (float): Activated output of the hidden layer.
        W2 (numpy.ndarray): Weights for the output layer.
        b2 (float): Bias for the output layer.
        A2 (float): Activated output of the output layer.
    """

    def __init__(self, nx, nodes):
        """
        Initializes a neural network.

        Args:
            nx (int): Number of input features.
            nodes (int): Number of nodes in the hidden layer.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
            TypeError: If nodes is not an integer.
            ValueError: If nodes is less than 1.
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
        """
        Getter for W1 (weights for the hidden layer).

        Returns:
            numpy.ndarray: Weights of the hidden layer.
        """
        return self.__W1

    @property
    def b1(self):
        """
        Getter for b1 (bias for the hidden layer).

        Returns:
            numpy.ndarray: Bias of the hidden layer.
        """
        return self.__b1

    @property
    def A1(self):
        """
        Getter for A1 (activated output of the hidden layer).

        Returns:
            float: Activation value of the hidden layer.
        """
        return self.__A1

    @property
    def W2(self):
        """
        Getter for W2 (weights for the output layer).

        Returns:
            numpy.ndarray: Weights of the output layer.
        """
        return self.__W2

    @property
    def b2(self):
        """
        Getter for b2 (bias for the output layer).

        Returns:
            float: Bias of the output layer.
        """
        return self.__b2

    @property
    def A2(self):
        """
        Getter for A2 (activated output of the output layer).

        Returns:
            float: Activation value of the output layer.
        """
        return self.__A2
