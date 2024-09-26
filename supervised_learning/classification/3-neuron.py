#!/usr/bin/env python3
import numpy as np

class Neuron:
    def __init__(self, nx):
        """
        Initializes the Neuron class.

        Parameters:
        nx (int): The number of input features.
        """
        if not isinstance(nx, int) or nx < 1:
            raise ValueError("nx must be a positive integer")
        
        self.__A = None  # Activated output
        self.__b = 0
        self.__W = np.random.randn(1, nx) * 0.01

    @property
    def W(self):
        """ Getter for the weights """
        return self.__W

    @property
    def b(self):
        """ Getter for the bias """
        return self.__b

    @b.setter
    def b(self, value):
        """ Setter for the bias """
        self.__b = value

    @property
    def A(self):
        """ Getter for the activated output """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.

        Parameters:
        X (numpy.ndarray): The input data with shape (nx, m).

        Returns:
        numpy.ndarray: The activated output.
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
        epsilon = 1e-15  # Use a very small epsilon to avoid log(0)
        A = np.clip(A, epsilon, 1 - epsilon)  # Clip values to avoid log(0)
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
        return cost
