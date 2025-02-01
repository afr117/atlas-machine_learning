#!/usr/bin/env python3
import numpy as np

class NeuralNetwork:
    """
    Class that defines a neural network with one hidden layer for binary classification.
    """

    def __init__(self, nx, nodes):
        """
        Initializes the neural network.
        nx: number of input features.
        nodes: number of nodes in the hidden layer.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize weights and biases
        self.W1 = np.random.randn(nodes, nx)  # Weights for the hidden layer
        self.b1 = np.zeros((nodes, 1))  # Bias for the hidden layer
        self.A1 = np.zeros((nodes, 1))  # Activated output for the hidden layer

        self.W2 = np.random.randn(1, nodes)  # Weights for the output layer
        self.b2 = np.zeros((1, 1))  # Bias for the output layer
        self.A2 = np.zeros((1, 1))  # Activated output for the output neuron

    def forward_prop(self, X):
        """
        Perform forward propagation to calculate activations.
        X: input data
        Returns A1 and A2: activations for hidden and output layers.
        """
        # Linear component for the hidden layer
        Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = 1 / (1 + np.exp(-Z1))  # Sigmoid activation for hidden layer

        # Linear component for the output layer
        Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation for output layer

        return self.A1, self.A2

    def cost(self, Y, A):
        """
        Compute the cost using binary cross-entropy.
        Y: true labels
        A: predicted output
        Returns the cost as a scalar value.
        """
        m = Y.shape[1]  # Number of examples

        # Cost function (no loops, vectorized calculation)
        cost = -np.mean(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the model's performance.
        X: input data (nx, m)
        Y: true labels (1, m)
        Returns predictions and the cost of the model.
        """
        # Perform forward propagation
        _, A = self.forward_prop(X)

        # Make predictions: 1 if A >= 0.5, otherwise 0 (vectorized)
        predictions = (A >= 0.5).astype(int)

        # Calculate the cost
        cost = self.cost(Y, A)

        return predictions, cost
