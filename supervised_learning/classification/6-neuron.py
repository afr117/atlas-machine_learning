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

    def cost(self, Y, A):
        """
        Computes the cost function for binary classification.
        Y: True labels
        A: Activated output
        Returns the cost value.
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
        return cost

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron.
        X: Input data
        Y: True labels
        iterations: Number of iterations for training
        alpha: Learning rate
        Returns the final activated output and the final cost.
        """
        # Validate iterations
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        # Validate alpha
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        # Perform training
        for i in range(iterations):
            A = self.forward_prop(X)  # Forward propagation
            self.gradient_descent(X, Y, A, alpha)  # Gradient descent
            if i % 1000 == 0:  # Print cost every 1000 iterations
                cost = self.cost(Y, A)
                print(f"Cost after {i} iterations: {cost}")

        # Return final activated output and final cost
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        return A, cost

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def evaluate(self, X, Y):
        """
        Evaluates the neuron after training.
        X: Input data
        Y: True labels
        Returns the activated output and the cost.
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        return A, cost
