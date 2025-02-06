#!/usr/bin/env python3

import numpy as np

class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        """
        Constructor for the DeepNeuralNetwork class.

        Args:
            nx (int): The number of input features.
            layers (list): A list of the number of nodes in each layer.

        Raises:
            TypeError: If nx is not an integer or layers is not a list of positive integers.
            ValueError: If nx is less than 1 or layers is an empty list.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        
        if not all(isinstance(i, int) and i > 0 for i in layers):
            raise TypeError("layers must be a list of positive integers")

        # Initialize public attributes
        self.L = len(layers)  # The number of layers in the network
        self.cache = {}  # A dictionary to hold all intermediary values
        self.weights = {}  # A dictionary to hold all weights and biases

        # Initialize weights and biases using He initialization
        for l in range(1, self.L + 1):
            if l == 1:
                # First layer, input to hidden layer
                self.weights[f'W{l}'] = np.random.randn(layers[l - 1], nx) * np.sqrt(2. / nx)
            else:
                # Subsequent layers
                self.weights[f'W{l}'] = np.random.randn(layers[l - 1], layers[l - 2]) * np.sqrt(2. / layers[l - 2])
            self.weights[f'b{l}'] = np.zeros((layers[l - 1], 1))  # Biases initialized to zero
