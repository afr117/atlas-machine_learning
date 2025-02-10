#!/usr/bin/env python3
"""
Defines a single neuron performing binary classification
"""
import numpy as np


class Neuron:
    """
    Class that defines a single neuron for binary classification
    """
    def __init__(self, nx):
        """
        Initializes the neuron
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        np.random.seed(0)  # Ensure deterministic initialization
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0
    
    @property
    def W(self):
        return self.__W
    
    @property
    def b(self):
        return self.__b
    
    @property
    def A(self):
        return self.__A
    
    def forward_prop(self, X):
        """
        Performs forward propagation using sigmoid activation function
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A
    
    def cost(self, Y, A):
        """
        Computes the cost using logistic regression loss
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
        return np.round(cost, decimals=10)
    
    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions
        """
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost
    
    def gradient_descent(self, X, Y, A, alpha):
        """
        Performs one pass of gradient descent
        """
        m = Y.shape[1]
        dW = np.matmul(A - Y, X.T) / m
        db = np.sum(A - Y) / m
        self.__W -= np.round(alpha * dW, decimals=10)
        self.__b -= np.round(alpha * db, decimals=10)
    
    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron using gradient descent
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        
        for _ in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
        
        return self.evaluate(X, Y)
