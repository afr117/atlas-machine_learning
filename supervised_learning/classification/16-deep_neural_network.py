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

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        def initialize_weights(index, prev_layer):
            if index > self.L:
                return
            self.weights[f"W{index}"] = (
                np.random.randn(layers[index - 1], prev_layer) * np.sqrt(2 / prev_layer)
            )
            self.weights[f"b{index}"] = np.zeros((layers[index - 1], 1))
            initialize_weights(index + 1, layers[index - 1])
        
        initialize_weights(1, nx)
