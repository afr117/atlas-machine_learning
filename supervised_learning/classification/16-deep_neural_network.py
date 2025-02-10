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
        if any(not isinstance(l, int) or l < 1 for l in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(1, self.L + 1):
            layer_size = layers[l - 1]
            prev_layer_size = nx if l == 1 else layers[l - 2]
            self.weights[f"W{l}"] = np.random.randn(layer_size, prev_layer_size) * np.sqrt(2 / prev_layer_size)
            self.weights[f"b{l}"] = np.zeros((layer_size, 1))
