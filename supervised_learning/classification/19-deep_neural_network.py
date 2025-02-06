#!/usr/bin/env python3

import numpy as np

class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        """
        Class constructor for a deep neural network with binary classification.

        Args:
        - nx (int): number of input features.
        - layers (list): a list of the number of nodes in each layer.
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
        self.L = len(layers)  # Number of layers
        self.cache = {}  # Dictionary to store intermediate values
        self.weights = {}  # Dictionary to store weights and biases
        
        previous_layer_nodes = nx  # Start with the input size for the first layer
        
        # Initialize weights and biases for all layers
        for l in range(self.L):
            # He initialization for weights
            self.weights[f'W{l + 1}'] = np.random.randn(layers[l], previous_layer_nodes) * np.sqrt(2. / previous_layer_nodes)
            # Initialize biases to zero
            self.weights[f'b{l + 1}'] = np.zeros((layers[l], 1))
            previous_layer_nodes = layers[l]  # Update the number of nodes for the next layer
    
    def forward_prop(self, X):
        """
        Performs forward propagation through the network.

        Args:
        - X (numpy.ndarray): Input data with shape (nx, m).

        Returns:
        - A (numpy.ndarray): Activated output of the last layer (1, m).
        - cache (dict): Dictionary containing all intermediate values.
        """
        A = X
        self.cache["A0"] = A  # Cache the input layer
        
        for l in range(1, self.L + 1):
            Z = np.dot(self.weights[f'W{l}'], A) + self.weights[f'b{l}']
            A = 1 / (1 + np.exp(-Z))  # Sigmoid activation function
            self.cache[f"A{l}"] = A  # Cache the output of each layer
            
        return A, self.cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Args:
        - Y (numpy.ndarray): True labels of shape (1, m).
        - A (numpy.ndarray): Predicted activated output of shape (1, m).

        Returns:
        - cost (float): The cost of the model.
        """
        m = Y.shape[1]  # Number of examples
        # Compute the binary cross-entropy cost
        cost = -np.mean(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost
