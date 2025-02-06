#!/usr/bin/env python3

import numpy as np

class Neuron:
    def __init__(self, nx):
        """
        Class constructor of a neuron with binary classification.

        Args:
        - nx (int): number of input features.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Initialize the parameters W (weights) and b (bias)
        self.__W = np.random.randn(1, nx)  # Weights initialization
        self.__b = 0  # Bias initialization
        self.__A = 0  # Activated output initialization

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def sigmoid(self, Z):
        """
        Sigmoid activation function.

        Args:
        - Z (numpy.ndarray): The input to the sigmoid function.

        Returns:
        - A (numpy.ndarray): The activated output.
        """
        return 1 / (1 + np.exp(-Z))

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Args:
        - Y (numpy.ndarray): True labels of shape (1, m).
        - A (numpy.ndarray): Predicted activated output of shape (1, m).

        Returns:
        - cost (float): The cost of the model.
        """
        m = Y.shape[1]  # Number of examples
        # Compute the binary cross-entropy cost
        cost = -np.mean(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron.

        Args:
        - X (numpy.ndarray): Input data with shape (nx, m).
        - Y (numpy.ndarray): True labels with shape (1, m).
        - iterations (int): Number of iterations of training.
        - alpha (float): Learning rate of gradient descent.

        Returns:
        - A (numpy.ndarray): Final output after training.
        - cost (float): Final cost after training.
        """
        m = X.shape[1]  # Number of examples
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        
        # Gradient descent loop (allowed to use one loop)
        for _ in range(iterations):
            # Forward propagation: Z = W * X + b, A = sigmoid(Z)
            Z = np.dot(self.__W, X) + self.__b
            self.__A = self.sigmoid(Z)
            
            # Compute the cost
            cost = self.cost(Y, self.__A)

            # Backward propagation: Compute gradients
            dZ = self.__A - Y  # Derivative of the cost w.r.t. Z
            dW = np.dot(dZ, X.T) / m  # Derivative w.r.t. W
            db = np.sum(dZ) / m  # Derivative w.r.t. b

            # Update parameters
            self.__W -= alpha * dW
            self.__b -= alpha * db

        return self.__A, cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron after training.

        Args:
        - X (numpy.ndarray): Input data with shape (nx, m).
        - Y (numpy.ndarray): True labels with shape (1, m).

        Returns:
        - A (numpy.ndarray): The activated output of the neuron.
        - cost (float): The cost of the model.
        """
        # Forward propagation
        A, cost = self.train(X, Y, iterations=0, alpha=0)  # Only get the forward output
        return A, cost
