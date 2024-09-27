#!/usr/bin/env python3
import numpy as np

class NeuralNetwork:
    """Defines a neural network with one hidden layer performing binary classification"""

    def __init__(self, nx, nodes):
        """Initialize the Neural Network"""
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

        # Initialize weights and biases for the hidden layer
        self.__W1 = np.random.randn(nodes, nx)  # Random normal distribution
        self.__b1 = np.zeros((nodes, 1))        # Zero-initialized biases
        self.__A1 = 0                           # Activated output initialized to zero
        
        # Initialize weights and biases for the output layer
        self.__W2 = np.random.randn(1, nodes)   # Random normal distribution
        self.__b2 = np.zeros((1, 1))            # Zero-initialized biases
        self.__A2 = 0                           # Activated output initialized to zero

    # Getter for W1
    @property
    def W1(self):
        return self.__W1
    
    # Getter for b1
    @property
    def b1(self):
        return self.__b1

    # Getter for A1
    @property
    def A1(self):
        return self.__A1

    # Getter for W2
    @property
    def W2(self):
        return self.__W2

    # Getter for b2
    @property
    def b2(self):
        return self.__b2

    # Getter for A2
    @property
    def A2(self):
        return self.__A2

