#!/usr/bin/env python3

import numpy as np


class NeuralNetwork:
    """
    A neural network with a single hidden layer, used for binary classification.
    The network performs forward propagation, cost calculation, and evaluation of predictions.
    """

    def __init__(self, nx, nodes):
        """
        Initializes the neural network by validating the input dimensions and setting up the weights.

        Args:
            nx (int): The number of input features for the neural network.
            nodes (int): The number of nodes in the hidden layer.

        Raises:
            TypeError: If nx or nodes is not an integer.
            ValueError: If nx or nodes is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize the weights, biases, and activated outputs for both layers
        self.W1 = np.random.randn(nodes, nx)  # Weights for the hidden layer
        self.b1 = np.zeros((nodes, 1))  # Bias for the hidden layer
        self.A1 = np.zeros((nodes, 1))  # Activated output for the hidden layer

        self.W2 = np.random.randn(1, nodes)  # Weights for the output layer
        self.b2 = np.zeros((1, 1))  # Bias for the output layer
        self.A2 = np.zeros((1, 1))  # Activated output for the output layer

    def forward_prop(self, X):
        """
        Perform forward propagation to compute the activations for both layers.

        Args:
            X (numpy.ndarray): The input data, shape (nx, m), where nx is the number of input features
                               and m is the number of examples.

        Returns:
            tuple: The activations for the hidden layer (A1) and the output layer (A2).
        """
        Z1 = np.dot(self.W1, X) + self.b1  # Linear transformation for the hidden layer
        self.A1 = 1 / (1 + np.exp(-Z1))  # Sigmoid activation function for the hidden layer

        Z2 = np.dot(self.W2, self.A1) + self.b2  # Linear transformation for the output layer
        self.A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation function for the output layer

        return self.A1, self.A2

    def cost(self, Y, A):
        """
        Calculate the cost using binary cross-entropy loss.

        Args:
            Y (numpy.ndarray): True labels for the input data, shape (1, m).
            A (numpy.ndarray): Predicted output of the neural network, shape (1, m).

        Returns:
            float: The binary cross-entropy cost of the model.
        """
        m = Y.shape[1]  # Number of examples

        # Compute the binary cross-entropy cost
        cost = -np.mean(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the neural network's predictions.

        Args:
            X (numpy.ndarray): The input data, shape (nx, m).
            Y (numpy.ndarray): The true labels, shape (1, m).

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: The predictions (1 for A >= 0.5, otherwise 0).
                - float: The cost of the model.
        """
        _, A = self.forward_prop(X)  # Perform forward propagation

        # Predictions: 1 if A >= 0.5, else 0 (vectorized)
        predictions = (A >= 0.5).astype(int)

        # Calculate the cost using the cost method
        cost = self.cost(Y, A)

        return predictions, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Performs one pass of gradient descent on the neural network.

        Args:
            X (numpy.ndarray): The input data, shape (nx, m).
            Y (numpy.ndarray): True labels for the input data, shape (1, m).
            A1 (numpy.ndarray): Activated output from the hidden layer, shape (nodes, m).
            A2 (numpy.ndarray): Activated output from the output layer, shape (1, m).
            alpha (float): The learning rate.

        Updates the weights and biases using gradient descent.
        """
        m = X.shape[1]  # Number of examples

        # Compute the derivative of the cost with respect to A2 (output layer)
        dz2 = A2 - Y  # Derivative of the cost with respect to A2
        dw2 = np.dot(dz2, A1.T) / m  # Derivative of the cost with respect to W2
        db2 = np.sum(dz2) / m  # Derivative of the cost with respect to b2

        # Compute the derivative of the cost with respect to A1 (hidden layer)
        dz1 = np.dot(self.W2.T, dz2) * A1 * (1 - A1)  # Derivative of the cost with respect to A1
        dw1 = np.dot(dz1, X.T) / m  # Derivative of the cost with respect to W1
        db1 = np.sum(dz1, axis=1, keepdims=True) / m  # Derivative of the cost with respect to b1

        # Update the weights and biases using gradient descent
        self.W2 -= alpha * dw2
        self.b2 -= alpha * db2
        self.W1 -= alpha * dw1
        self.b1 -= alpha * db1
