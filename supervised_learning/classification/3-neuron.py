#!/usr/bin/env python3
"""
3-neuron.py
Defines a single neuron performing binary classification.
"""

import numpy as np

class Neuron:
    """
    Class that defines a single neuron performing binary classification.
    """

    def __init__(self, nx):
        """
        Initialize a Neuron instance.

        Parameters:
        nx (int): Number of input features to the neuron.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        self.__A = 0
        self.__b = 0
        self.__W = np.random.randn(1, nx)
    
    @property
    def A(self):
        """
        Getter for the private attribute __A.
        """
        return self.__A

    @property
    def b(self):
        """
        Getter for the private attribute __b.
        """
        return self.__b

    @b.setter
    def b(self, value):
        """
        Setter for the private attribute __b.
        """
        self.__b = value

    @property
    def W(self):
        """
        Getter for the private attribute __W.
        """
        return self.__W

    @W.setter
    def W(self, value):
        """
        Setter for the private attribute __W.
        """
        self.__W = value

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.

        Parameters:
        X (numpy.ndarray): Input data with shape (nx, m).

        Returns:
        numpy.ndarray: The activated output of the neuron.
        """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Parameters:
        Y (numpy.ndarray): Correct labels for the input data with shape (1, m).
        A (numpy.ndarray): Activated output of the neuron with shape (1, m).

        Returns:
        float: The cost of the model.
        """
        m = Y.shape[1]
        # Use a small epsilon value to prevent division by zero
        epsilon = 1e-10
        A = np.clip(A, epsilon, 1 - epsilon)  # Clip values to avoid log(0)
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
        return cost
