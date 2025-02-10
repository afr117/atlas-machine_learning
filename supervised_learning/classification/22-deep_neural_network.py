#!/usr/bin/env python3
import numpy as np


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

        def initialize_weights(index, prev_layer):
            if index > self.__L:
                return
            self.__weights[f"W{index}"] = (
                np.random.randn(layers[index - 1], prev_layer) * np.sqrt(2 / prev_layer)
            )
            self.__weights[f"b{index}"] = np.zeros((layers[index - 1], 1))
            initialize_weights(index + 1, layers[index - 1])
        
        initialize_weights(1, nx)

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
        """Calculates forward propagation of the deep neural network"""
        self.__cache["A0"] = X
        def activate(layer):
            if layer > self.__L:
                return self.__cache[f"A{self.__L}"]
            W = self.__weights[f"W{layer}"]
            b = self.__weights[f"b{layer}"]
            Z = np.matmul(W, self.__cache[f"A{layer - 1}"]) + b
            self.__cache[f"A{layer}"] = 1 / (1 + np.exp(-Z))
            return activate(layer + 1)
        return activate(1), self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.where(A >= 0.5, 1, 0)
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Performs one pass of gradient descent on the neural network"""
        m = Y.shape[1]
        dZ = cache[f"A{self.__L}"] - Y
        
        def update_weights(layer, dZ):
            if layer < 1:
                return
            A_prev = cache[f"A{layer - 1}"]
            W = self.__weights[f"W{layer}"]
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            self.__weights[f"W{layer}"] -= alpha * dW
            self.__weights[f"b{layer}"] -= alpha * db
            
            if layer > 1:
                dZ = np.matmul(W.T, dZ) * (A_prev * (1 - A_prev))
                update_weights(layer - 1, dZ)
        
        update_weights(self.__L, dZ)
    
    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Trains the deep neural network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step < 1 or step > iterations:
            raise ValueError("step must be positive and <= iterations")

        costs = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)

            if verbose and i % step == 0:
                print(f"Cost after {i} iterations: {cost}")

            if graph and i % step == 0:
                costs.append((i, cost))

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            x_vals, y_vals = zip(*costs)
            plt.plot(x_vals, y_vals, 'b-')
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
