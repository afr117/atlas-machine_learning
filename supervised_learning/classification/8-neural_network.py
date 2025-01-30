#!/usr/bin/env python3
import numpy as np

class NeuralNetwork:
    def __init__(self, nx, nodes):
        """
        Constructor to initialize the neural network with one hidden layer.
        nx: The number of input features
        nodes: The number of nodes in the hidden layer
        """
        # Validate nx (number of input features)
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate nodes (number of nodes in the hidden layer)
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize the neural network's parameters (weights and biases)
        self.W1 = np.random.randn(nodes, nx)  # Weights for the hidden layer (randomly initialized)
        self.b1 = np.zeros((nodes, 1))  # Bias for the hidden layer (initialized to 0)
        self.A1 = np.zeros((nodes, 1))  # Activated output for the hidden layer (initialized to 0)
        
        self.W2 = np.random.randn(1, nodes)  # Weights for the output layer (randomly initialized)
        self.b2 = np.zeros((1, 1))  # Bias for the output layer (initialized to 0)
        self.A2 = np.zeros((1, 1))  # Activated output for the output neuron (initialized to 0)
