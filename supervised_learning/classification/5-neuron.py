#!/usr/bin/env python3
import numpy as np

class Neuron:
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        self.__W = np.random.randn(1, nx)  # Weights initialized randomly
        self.__b = 0  # Bias initialized to 0
        self.__A = 0  # Activated output initialized to 0
    
    def forward_prop(self, X):
        """
        Performs forward propagation using the sigmoid activation function.
        X: The input data.
        Returns the activated output A.
        """
        Z = np.dot(self.__W, X) + self.__b  # Linear transformation
        self.__A = 1 / (1 + np.exp(-Z))  # Sigmoid activation function
        return self.__A

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Performs one pass of gradient descent on the neuron.
        X: Input data
        Y: True labels
        A: Activated output from forward propagation
        alpha: Learning rate
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
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A
