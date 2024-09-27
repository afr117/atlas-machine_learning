#!/usr/bin/env python3
import numpy as np


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer performing binary classification.
    """

    def __init__(self, nx, nodes):
        """
        Initializes the NeuralNetwork.

        Args:
        nx: number of input features (must be a positive integer).
        nodes: number of nodes in the hidden layer (must be a positive integer).
        """
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate nodes
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize the private weights and biases for the hidden layer
        self.__W1 = np.random.randn(nodes, nx)  # Random normal distribution
        self.__b1 = np.zeros((nodes, 1))        # Zero-initialized biases
        self.__A1 = 0                           # Activated output initially zero

        # Initialize the private weights and biases for the output layer
        self.__W2 = np.random.randn(1, nodes)   # Random normal distribution
        self.__b2 = np.zeros((1, 1))            # Zero-initialized biases
        self.__A2 = 0                           # Activated output initially zero

    @property
    def W1(self):
        """Getter for W1 (weights of the hidden layer)."""
        return self.__W1

    @property
    def b1(self):
        """Getter for b1 (bias of the hidden layer)."""
        return self.__b1

    @property
    def A1(self):
        """Getter for A1 (activated output of the hidden layer)."""
        return self.__A1

    @property
    def W2(self):
        """Getter for W2 (weights of the output layer)."""
        return self.__W2

    @property
    def b2(self):
        """Getter for b2 (bias of the output layer)."""
        return self.__b2

    @property
    def A2(self):
        """Getter for A2 (activated output of the output layer)."""
        return self.__A2

