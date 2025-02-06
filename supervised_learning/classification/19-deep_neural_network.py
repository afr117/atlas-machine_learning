#!/usr/bin/env python3

import numpy as np

class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        # Same constructor as the previous task
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(i, int) and i > 0 for i in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        
        previous_layer_nodes = nx
        for l in range(self.L):
            self.weights[f'W{l + 1}'] = np.random.randn(layers[l], previous_layer_nodes) * np.sqrt(2. / previous_layer_nodes)
            self.weights[f'b{l + 1}'] = np.zeros((layers[l], 1))
            previous_layer_nodes = layers[l]
    
    def forward_prop(self, X):
        # Forward propagation method (same as in the previous task)
        A = X
        self.cache["A0"] = A  # Cache the input layer
        
        for l in range(1, self.L + 1):
            Z = np.dot(self.weights[f'W{l}'], A) + self.weights[f'b{l}']
            A = 1 / (1 + np.exp(-Z))  # Sigmoid activation function
            self.cache[f"A{l}"] = A  # Cache activated output of the current layer
            
        return A, self.cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression (binary cross-entropy).
        
        Args:
            Y (numpy.ndarray): The true labels (1, m).
            A (numpy.ndarray): The activated output (1, m).
        
        Returns:
            float: The cost of the model.
        """
        m = Y.shape[1]  # Number of examples
        # Compute the binary cross-entropy cost
        cost = -np.mean(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost
