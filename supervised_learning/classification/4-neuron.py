#!/usr/bin/env python3
"""
Module 4-neuron
Defines a class Neuron for binary classification.
"""

import numpy as np


class Neuron:
    """
    Class Neuron that defines a single neuron performing binary classification.
    """

    def __init__(self, nx):
        """
        Initialize a neuron.
        
        Args:
            nx (int): The number of input features to the neuron.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        np.random.seed(0)  # Ensure deterministic weight initialization
        self.W = np.random.randn(1, nx)  # Weight initialization using numpy
        self.b = 0  # Bias initialization
        self.A = 0  # Activated output placeholder

    def sigmoid(self, z):
        """
        Compute the sigmoid activation function.
        
        Args:
            z (numpy.ndarray): The input data to the sigmoid function.
        
        Returns:
            numpy.ndarray: The sigmoid activation.
        """
        return 1 / (1 + np.exp(-z))

    def forward_prop(self, X):
        """
        Perform forward propagation of the neuron.
        
        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
        
        Returns:
            numpy.ndarray: The activated output of the neuron.
        """
        z = np.dot(self.W, X) + self.b  # Linear transformation of the inputs
        self.A = self.sigmoid(z)  # Apply sigmoid activation function
        return self.A

    def cost(self, Y, A):
        """
        Calculate the cost using binary cross-entropy.
        
        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m).
            A (numpy.ndarray): Activated output with shape (1, m).
        
        Returns:
            float: The cost of the model.
        """
        m = Y.shape[1]
        # Add epsilon to A to avoid log(0) issues
        epsilon = 1e-8
        cost = -(1 / m) * np.sum(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the neuron’s predictions.
        
        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m).
        
        Returns:
            numpy.ndarray: The predicted labels for each example.
            float: The cost of the network.
        """
        A = self.forward_prop(X)  # Perform forward propagation
        cost = self.cost(Y, A)  # Compute the cost
        # Convert the predicted probabilities to binary labels (0 or 1)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost
