#!/usr/bin/env python3

import numpy as np
import pickle

class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Class constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if any(map(lambda l: not isinstance(l, int) or l <= 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(1, self.__L + 1):
            self.__weights[f"W{i}"] = (
                np.random.randn(layers[i - 1], nx if i == 1 else layers[i - 2]) * np.sqrt(2 / (nx if i == 1 else layers[i - 2]))
            )
            self.__weights[f"b{i}"] = np.zeros((layers[i - 1], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Performs forward propagation"""
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            Z = np.matmul(self.__weights[f"W{i}"], self.__cache[f"A{i-1}"]) + self.__weights[f"b{i}"]
            self.__cache[f"A{i}"] = 1 / (1 + np.exp(-Z))
        return self.__cache[f"A{self.__L}"], self.__cache

    def cost(self, Y, A):
        """Calculates the cost using logistic regression"""
        m = Y.shape[1]
        return -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.where(A >= 0.5, 1, 0)
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Performs one pass of gradient descent"""
        m = Y.shape[1]
        dZ = cache[f"A{self.__L}"] - Y

        for i in range(self.__L, 0, -1):
            A_prev = cache[f"A{i-1}"]
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            self.__weights[f"W{i}"] -= alpha * dW
            self.__weights[f"b{i}"] -= alpha * db

            if i > 1:
                dZ = np.matmul(self.__weights[f"W{i}"].T, dZ) * (A_prev * (1 - A_prev))

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Trains the deep neural network"""
        if not isinstance(iterations, int) or iterations < 1:
            raise TypeError("iterations must be an integer") if not isinstance(iterations, int) else ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float) or alpha <= 0:
            raise TypeError("alpha must be a float") if not isinstance(alpha, float) else ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int) or step <= 0 or step > iterations:
                raise TypeError("step must be an integer") if not isinstance(step, int) else ValueError("step must be positive and <= iterations")

        costs = []
        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            if verbose and i % step == 0:
                cost = self.cost(Y, A)
                costs.append(cost)
                print(f"Cost after {i} iterations: {cost}")
        if graph:
            plt.plot(range(0, iterations, step), costs, 'b-')
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
