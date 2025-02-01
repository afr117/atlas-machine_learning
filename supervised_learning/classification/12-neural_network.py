#!/usr/bin/env python3
import numpy as np

class NeuralNetwork:
    def __init__(self, nx, nodes):
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
        
        # Initialize parameters (weights and biases)
        self.W1 = np.random.randn(nodes, nx)  # Weights for the hidden layer
        self.b1 = np.zeros((nodes, 1))  # Bias for the hidden layer
        self.A1 = np.zeros((nodes, 1))  # Activated output for the hidden layer
        
        self.W2 = np.random.randn(1, nodes)  # Weights for the output layer
        self.b2 = np.zeros((1, 1))  # Bias for the output layer
        self.A2 = np.zeros((1, 1))  # Activated output for the output neuron

    def forward_prop(self, X):
        """
        Perform forward propagation and compute the activations.
        """
        # Z1 is the linear component for the hidden layer
        Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = 1 / (1 + np.exp(-Z1))  # Sigmoid activation for hidden layer
        
        # Z2 is the linear component for the output layer
        Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation for output layer
        
        return self.A1, self.A2

    def cost(self, Y, A):
        """
        Calculate the cost using logistic regression.
        Y is the true labels, A is the predicted labels (activated output).
        """
        m = Y.shape[1]  # Number of examples

        # Vectorized cost calculation: No loops here
        cost = -np.mean(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the neural network’s predictions.
        X is the input data (nx, m).
        Y is the true labels (1, m).
        Returns the predictions and the cost.
        """
        # Perform forward propagation to get the activated output A2
        _, A = self.forward_prop(X)
        
        # Make predictions: 1 if output >= 0.5, otherwise 0
        predictions = (A >= 0.5).astype(int)  # Vectorized prediction
        
        # Calculate the cost using the cost method (also vectorized)
        cost = self.cost(Y, A)
        
        return predictions, cost
