#!/usr/bin/env python3
import numpy as np

class NeuralNetwork:
    """
    A neural network with a single hidden layer, used for binary classification.
    """

    def __init__(self, nx, nodes):
        """
        Initializes the neural network by validating the input dimensions and setting up the weights.
        nx: the number of input features
        nodes: the number of nodes in the hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize the weights and biases
        self.W1 = np.random.randn(nodes, nx)  # Weights for the hidden layer
        self.b1 = np.zeros((nodes, 1))  # Bias for the hidden layer
        self.A1 = np.zeros((nodes, 1))  # Activated output for the hidden layer

        self.W2 = np.random.randn(1, nodes)  # Weights for the output layer
        self.b2 = np.zeros((1, 1))  # Bias for the output layer
        self.A2 = np.zeros((1, 1))  # Activated output for the output neuron

    def forward_prop(self, X):
        """
        Perform forward propagation to compute the activations.
        X: input data
        Returns the activations for the hidden and output layers
        """
        Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = 1 / (1 + np.exp(-Z1))  # Sigmoid activation for the hidden layer

        Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation for the output layer

        return self.A1, self.A2

    def cost(self, Y, A):
        """
        Calculate the cost using binary cross-entropy loss.
        Y: true labels
        A: predicted output
        Returns the cost as a scalar value
        """
        m = Y.shape[1]  # Number of examples

        cost = -np.mean(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the neural network's predictions.
        X: input data
        Y: true labels
        Returns the predictions and the cost of the model
        """
        _, A = self.forward_prop(X)  # Perform forward propagation

        # Predictions: 1 if A >= 0.5, else 0 (vectorized)
        predictions = (A >= 0.5).astype(int)

        # Calculate the cost using the cost method
        cost = self.cost(Y, A)

        return predictions, cost
