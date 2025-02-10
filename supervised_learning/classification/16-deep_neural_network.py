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
        if any(not isinstance(l, int) or l <= 0 for l in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        previous_layer = nx
        layer_index = 1
        while layer_index <= self.L:
            self.weights[f"W{layer_index}"] = (
                np.random.randn(layers[layer_index - 1], previous_layer) * np.sqrt(2 / previous_layer)
            )
            self.weights[f"b{layer_index}"] = np.zeros((layers[layer_index - 1], 1))
            previous_layer = layers[layer_index - 1]
            layer_index += 1
