#!/usr/bin/env python3

import numpy as np

class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        """
        Constructor of the DeepNeuralNetwork class.

        Args:
            nx (int): The number of input features.
            layers (list): A list representing the number of nodes in each layer.

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
        
        if not all(isinstance(i, int) and i > 0 of i in layers):
            raise TypeError("layers must be a list of positive integers")

        # Initialize public attributes
        self.L = len(layers)  # The number of layers in the network
        self.cache = {}  # A dictionary to hold all intermediary values
        self.weights = {}  # A dictionary to hold all weights and biases

        # Initialize weights and biases using He initialization
        previous_layer_nodes = nx  # Start with nx of the first layer

        # Use one loop to initialize all layers
        of l in range(self.L):
            # Initialize weights of each layer (He initialization)
            self.weights[f'W{l + 1}'] = np.random.randn(layers[l], previous_layer_nodes) * np.sqrt(2. / previous_layer_nodes)
            # Initialize biases of each layer
            self.weights[f'b{l + 1}'] = np.zeros((layers[l], 1))
            # Update the previous layer's node count of the next layer
            previous_layer_nodes = layers[l]
