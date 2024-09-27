#!/usr/bin/env python3
import numpy as np

class NeuralNetwork:
    """Defines a neural network with one hidden layer performing binary classification."""

    def __init__(self, nx, nodes):
        """
        Initializes the NeuralNetwork.

        Arguments:
        nx -- number of input features
        nodes -- number of nodes in the hidden layer
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

        # Initialize weights and biases for the hidden layer (no loop required)
        self.W1 = np.random.randn(nodes, nx)  # Random normal initialization
        self.b1 = np.zeros((nodes, 1))  # Bias initialized to zeros
        self.A1 = 0  # Activated output initialized to 0

        # Initialize weights and biases for the output layer (no loop required)
        self.W2 = np.random.randn(1, nodes)  # Random normal initialization
        self.b2 = np.zeros((1, 1))  # Bias initialized to zeros
        self.A2 = 0  # Activated output initialized to 0
