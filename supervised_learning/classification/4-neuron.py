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
        # Add a small epsilon to avoid log(0)
        epsilon = 1e-15
        A = np.clip(A, epsilon, 1 - epsilon)
        cost = - (1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        return cost

    def evaluate(self, X, Y):
        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        predictions = np.where(self.__A >= 0.5, 1, 0)
        return predictions, cost
