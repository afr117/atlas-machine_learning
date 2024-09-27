#!/usr/bin/env python3
import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden layer performing binary classification"""

    def __init__(self, nx, nodes):
        """
        Class constructor
        Args:
            nx (int): Number of input features
            nodes (int): Number of nodes in the hidden layer
        Raises:
            TypeError: If nx is not an integer
            ValueError: If nx is less than 1
            TypeError: If nodes is not an integer
            ValueError: If nodes is less than 1
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

        # Initialize weights and biases for the hidden layer
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0

        # Initialize weights and biases for the output layer
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
