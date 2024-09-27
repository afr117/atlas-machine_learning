#!/usr/bin/env python3
import numpy as np

class NeuralNetwork:
    """
    Defines a neural network with one hidden layer performing binary classification.
    """
    
    def __init__(self, nx, nodes):
        """
        Initializes the NeuralNetwork.

        Args:
        nx: number of input features (must be a positive integer).
        nodes: number of nodes in the hidden layer (must be a positive integer).
        """
        
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        # Validate nodes
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        
        # Initialize the weights and biases for the hidden layer
        self.W1 = np.random.randn(nodes, nx)  # Random normal distribution
        self.b1 = np.zeros((nodes, 1))        # Zero-initialized biases
        self.A1 = 0                           # Activated output initially zero
        
        # Initialize the weights and biases for the output layer
        self.W2 = np.random.randn(1, nodes)   # Random normal distribution
        self.b2 = np.zeros((1, 1))            # Zero-initialized biases
        self.A2 = 0                           # Activated output initially zero

