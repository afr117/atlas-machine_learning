#!/usr/bin/env python3

import numpy as np


class Neuron:
    """
    A class that represents a single neuron performing binary classification.
    """

    def __init__(self, nx):
        """
        Initializes the neuron with the given number of input features.

        Parameters:
        nx (int): The number of input features to the neuron.
        """
        if not isinstance(nx, int) or nx <= 0:
            raise ValueError("nx must be a positive integer")
        
        self.__A = np.zeros((1, 1))
        self.__b = 0
        self.__W = np.zeros((1, nx))
        self.__m = 0

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.

        Parameters:
        X (numpy.ndarray): Input data with shape (nx, m), where nx is the number
                           of input features and m is the number of examples.

        Returns:
        numpy.ndarray: The activation value of the neuron.
        """
        self.__m = X.shape[1]
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the neuron’s predictions.

        Parameters:
        Y (numpy.ndarray): Correct labels with shape (1, m).
        A (numpy.ndarray): Predicted labels with shape (1, m).

        Returns:
        float: The cost of the network.
        """
        m = Y.shape[1]
        cost = (-1 / m) * (np.matmul(Y, np.log(A).T) + np.matmul(1 - Y, np.log(1 - A).T))
        return np.squeeze(cost)

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions and calculates the cost.

        Parameters:
        X (numpy.ndarray): Input data with shape (nx, m).
        Y (numpy.ndarray): Correct labels with shape (1, m).

        Returns:
        tuple: (numpy.ndarray, float) where the first element is the predictions
               and the second element is the cost of the network.
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.where(A >= 0.5, 1, 0)
        return predictions, cost
