#!/usr/bin/env python3
import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """
        Initialize a Neuron

        nx: int, number of input features to the neuron
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Weight initialization using a normal distribution
        np.random.seed(0)
        self.W = np.random.randn(1, nx)
        self.b = 0  # Bias initialized to 0
        self.A = 0  # Activated output initialized to 0

    def sigmoid(self, z):
        """
        Sigmoid activation function
        
        z: numpy.ndarray, linear transformation of input data

        Returns: activated output
        """
        return 1 / (1 + np.exp(-z))

    def forward_prop(self, X):
        """
        Perform forward propagation

        X: numpy.ndarray (nx, m), input data
        nx: number of input features
        m: number of examples

        Returns: The activated output
        """
        z = np.dot(self.W, X) + self.b  # Linear transformation
        self.A = self.sigmoid(z)  # Activation using sigmoid
        return self.A

    def cost(self, Y, A):
        """
        Calculate the cost using logistic regression

        Y: numpy.ndarray (1, m), correct labels
        A: numpy.ndarray (1, m), activated output

        Returns: The cost
        """
        m = Y.shape[1]
        epsilon = 1e-8  # Small constant to avoid log(0)
        cost = -(1 / m) * np.sum(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the neuron's predictions

        X: numpy.ndarray (nx, m), input data
        Y: numpy.ndarray (1, m), correct labels

        Returns: The neuron's predictions and the cost of the model
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.where(A >= 0.5, 1, 0)  # Convert probabilities to binary labels
        return predictions, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Perform one pass of gradient descent on the neuron

        X: numpy.ndarray (nx, m), input data
        Y: numpy.ndarray (1, m), correct labels
        A: numpy.ndarray (1, m), activated output
        alpha: float, learning rate
        """
        m = X.shape[1]
        dZ = A - Y  # Derivative of the cost with respect to A
        dW = np.dot(dZ, X.T) / m  # Gradient of the cost with respect to W
        db = np.sum(dZ) / m  # Gradient of the cost with respect to b

        # Update the weights and bias
        self.W -= alpha * dW
        self.b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Train the neuron over a number of iterations

        X: numpy.ndarray (nx, m), input data
        Y: numpy.ndarray (1, m), correct labels
        iterations: int, number of iterations to train over
        alpha: float, learning rate

        Returns: The evaluation of the training data after training
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A = self.forward_prop(X)  # Forward propagation
            self.gradient_descent(X, Y, A, alpha)  # Gradient descent

        # Final evaluation after training
        return self.evaluate(X, Y)
