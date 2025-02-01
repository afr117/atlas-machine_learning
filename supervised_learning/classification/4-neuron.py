#!/usr/bin/env python3
import numpy as np

class Neuron:
    """
    A class representing a single neuron for binary classification.
    """
    def __init__(self, nx):
        """
        Initialize a Neuron instance.
        
        Args:
            nx (int): The number of input features to the neuron.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Initialize weights, bias, and activation output
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for the weights."""
        return self.__W

    @property
    def b(self):
        """Getter for the bias."""
        return self.__b

    @property
    def A(self):
        """Getter for the activation output."""
        return self.__A

    @A.setter
    def A(self, value):
        """Prevent setting the activation output attribute."""
        raise AttributeError("can't set attribute")

    def forward_prop(self, X):
        """
        Calculate the forward propagation of the neuron.
        
        Args:
            X (numpy.ndarray): The input data, shape (nx, m), where nx is
                               the number of input features and m is the number of examples.
        Returns:
            numpy.ndarray: The activated output of the neuron.
        """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))  # Sigmoid activation function
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        
        Args:
            Y (numpy.ndarray): True labels for the input data, shape (1, m).
            A (numpy.ndarray): Activated output of the neuron, shape (1, m).
        
        Returns:
            float: The cost of the model.
        """
        m = Y.shape[1]  # Number of examples
        # Compute the cost using the logistic regression formula
        cost = - (1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions.
        
        Args:
            X (numpy.ndarray): The input data, shape (nx, m).
            Y (numpy.ndarray): The correct labels, shape (1, m).
        
        Returns:
            numpy.ndarray: The predicted labels, shape (1, m).
            float: The cost of the model.
        """
        A = self.forward_prop(X)  # Get the activated output
        cost = self.cost(Y, A)  # Calculate the cost
        prediction = (A >= 0.5).astype(int)  # Convert A to binary predictions
        return prediction, cost
