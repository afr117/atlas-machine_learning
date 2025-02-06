#!/usr/bin/env python3

"""
This module defines a class Neuron that represents a
single neuron for binary classification.
The class allows for forward propagation using the
sigmoid activation function and performs
gradient descent to optimize the neuron’s weights and bias.
"""

import numpy as np


class Neuron:
    """
    A class representing a single neuron for binary classification.
    The neuron computes forward propagation using the
    sigmoid activation function and
    performs gradient descent to optimize the model.
    """

    def __init__(self, nx):
        """
        Initializes the Neuron instance with random weights, a bias,
        and sets the activation output to 0.

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
        
        # Initialize weights, bias, and activation output
        self.__W = np.random.randn(1, nx)  # Weights initialized randomly
        self.__b = 0  # Bias initialized to 0
        self.__A = 0  # Activated output initialized to 0
    
    def forward_prop(self, X):
        """
        Performs forward propagation using the sigmoid activation function.

        Args:
            X (numpy.ndarray): The input data, shape (nx, m),
            where nx is the number of input features
                               and m is the number of examples.

        Returns:
            numpy.ndarray: The activated output of the neuron after
            applying the sigmoid function.
        """
        Z = np.dot(self.__W, X) + self.__b  # Linear transformation
        self.__A = 1 / (1 + np.exp(-Z))  # Sigmoid activation function
        return self.__A

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Performs one pass of gradient descent on the neuron to
        optimize the weights and bias.

        Args:
            X (numpy.ndarray): The input data, shape (nx, m).
            Y (numpy.ndarray): The true labels, shape (1, m).
            A (numpy.ndarray): The activated output from forward propagation,
            shape (1, m).
            alpha (float, optional): The learning rate. Default is 0.05.
        
        Updates:
            The weights and bias are updated using the
            gradient descent rule.
        """
        m = X.shape[1]  # Number of examples

        # Calculate the derivative of the cost with respect to weights and bias
        dz = A - Y  # Derivative of the cost with respect to A
        dw = np.dot(dz, X.T) / m  # Derivative of the cost with respect to W
        db = np.sum(dz) / m  # Derivative of the cost with respect to b

        # Update the weights and bias using the gradient descent rule
        self.__W -= alpha * dw
        self.__b -= alpha * db

    @property
    def W(self):
        """
        Getter for the weights.

        Returns:
            numpy.ndarray: The weights of the neuron.
        """
        return self.__W

    @property
    def b(self):
        """
        Getter for the bias.

        Returns:
            float: The bias of the neuron.
        """
        return self.__b

    @property
    def A(self):
        """
        Getter for the activation output.

        Returns:
            numpy.ndarray: The activation output of the neuron.
        """
        return self.__A
