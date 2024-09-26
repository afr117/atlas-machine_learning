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
        # To avoid division by zero errors, use 1.0000001 - A instead of 1 - A
        A = np.clip(A, 1e-7, 1 - 1e-7)  # To avoid log(0) and improve stability
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions.

        Parameters:
        X (numpy.ndarray): The input data with shape (nx, m).
        Y (numpy.ndarray): Correct labels for the input data with shape (1, m).

        Returns:
        tuple: The neuron’s prediction and the cost of the network.
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = (A >= 0.5).astype(int)
        return prediction, cost
