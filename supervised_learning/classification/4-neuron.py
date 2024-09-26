#!/usr/bin/env python3
import numpy as np

class Neuron:
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """Initialize the neuron"""
        if not isinstance(nx, int) or nx <= 0:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)  # Weights initialized with normal distribution
        self.__b = 0  # Bias initialized to 0
        self.__A = 0  # Activated output (sigmoid) initialized to 0

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        Z = np.dot(self.__W, X) + self.__b  # Linear transformation
        self.__A = 1 / (1 + np.exp(-Z))  # Sigmoid activation
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost using logistic regression"""
        m = Y.shape[1]  # Number of examples
        epsilon = 1e-10  # Small value to avoid log(0)
        cost = -np.sum(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron's predictions and calculates the cost"""
        A = self.forward_prop(X)  # Forward propagation
        predictions = (A >= 0.5).astype(int)  # Threshold the predictions
        cost = self.cost(Y, A)  # Calculate the cost
        return predictions, cost

    @property
    def W(self):
        """Getter for weights"""
        return self.__W

    @property
    def b(self):
        """Getter for bias"""
        return self.__b

    @property
    def A(self):
        """Getter for activated output"""
        return self.__A
