#!/usr/bin/env python3

import numpy as np

class Neuron:
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    def forward_prop(self, X):
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A + 1e-15) + (1 - Y) * np.log(1 - A + 1e-15)) / m
        return cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        m = X.shape[1]
        dz = A - Y
        dw = np.dot(dz, X.T) / m
        db = np.sum(dz) / m
        self.__W -= alpha * dw
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        m = X.shape[1]
        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
        cost = self.cost(Y, A)
        return A, cost

    def evaluate(self, X, Y):
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.round(A)
        return predictions, cost

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A
