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
        if any(map(lambda layer_size: not isinstance(layer_size, int) or
                   layer_size <= 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        def initialize_weights(index, prev_layer):
            """Recursive weight initialization"""
            if index > self.__L:
                return
            self.__weights[f"W{index}"] = (
                np.random.randn(layers[index - 1], prev_layer)
                * np.sqrt(2 / prev_layer)
            )
            self.__weights[f"b{index}"] = np.zeros((layers[index - 1], 1))
            initialize_weights(index + 1, layers[index - 1])

        initialize_weights(1, nx)

    @property
    def L(self):
        """Getter for L"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates forward propagation of the deep neural network"""
        self.__cache["A0"] = X

        def activate(layer):
            """Recursive activation function"""
            if layer > self.__L:
                return self.__cache[f"A{self.__L}"]
            W = self.__weights[f"W{layer}"]
            b = self.__weights[f"b{layer}"]
            Z = np.matmul(W, self.__cache[f"A{layer - 1}"]) + b
            self.__cache[f"A{layer}"] = 1 / (1 + np.exp(-Z))
            return activate(layer + 1)

        return activate(1), self.__cache
