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
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize weights and biases for the hidden layer
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # Initialize weights and biases for the output layer
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = np.zeros((1, 1))
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def sigmoid(self, Z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Args:
        X: numpy.ndarray with shape (nx, m) that contains the input data.

        Returns:
        The activated outputs for the hidden layer (__A1) and the output layer (__A2).
        """
        # Calculate Z1 = W1 * X + b1
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = self.sigmoid(Z1)  # Apply the sigmoid activation function

        # Calculate Z2 = W2 * A1 + b2
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = self.sigmoid(Z2)  # Apply the sigmoid activation function

        # Return both activated outputs
        return self.__A1, self.__A2

