#!/usr/bin/env python3

import numpy as np


class Neuron:
    def __init__(self, nx):
        """Initialize the neuron with weights, bias, and activation."""
        if not isinstance(nx, int) or nx <= 0:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(1, nx)  # weights initialization
        self.b = 0  # bias initialization
        self.A = 0  # activation initialization

    def forward_prop(self, X):
        """Calculate the forward propagation of the neuron."""
        Z = np.dot(self.W, X) + self.b
        self.A = 1 / (1 + np.exp(-Z))  # Sigmoid activation
        return self.A, Z

    def cost(self, Y, A):
        """Calculate the cost using binary cross-entropy."""
        m = Y.shape[1]  # number of examples
        cost = - (1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluate the neuron’s predictions and cost."""
        A, _ = self.forward_prop(X)  # Get activation output
        cost = self.cost(Y, A)  # Calculate cost
        predictions = (A >= 0.5).astype(int)  # Convert probabilities to binary predictions
        return predictions, cost
