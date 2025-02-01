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
        """Performs forward propagation"""
        # Z1 is the linear component for the hidden layer
        Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = 1 / (1 + np.exp(-Z1))  # Sigmoid activation
        
        # Z2 is the linear component for the output layer
        Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation
        
        return self.A1, self.A2

    def cost(self, Y, A2):
        """Calculates the binary cross-entropy cost"""
        m = Y.shape[1]  # Number of examples
        cost = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluates the predictions and cost"""
        self.A1, self.A2 = self.forward_prop(X)
        cost = self.cost(Y, self.A2)
        predictions = (self.A2 >= 0.5).astype(int)  # Convert probabilities to binary predictions
        accuracy = np.mean(predictions == Y) * 100  # Accuracy as percentage
        return predictions, cost, accuracy

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=False, graph=False, step=100):
        """Trains the model using gradient descent"""
        
        # Validate inputs for training
        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float) or alpha <= 0:
            raise ValueError("alpha must be positive")
        if not isinstance(step, int) or step <= 0 or step > iterations:
            raise ValueError("step must be positive and <= iterations")
        
        m = X.shape[1]  # Number of examples
        costs = np.zeros(iterations)  # To store cost history
        
        # Training loop (using vectorized operations, no explicit loops)
        for i in range(iterations):
            # Forward propagation
            self.A1, self.A2 = self.forward_prop(X)
            
            # Compute gradients
            dZ2 = self.A2 - Y  # Derivative of cost with respect to A2
            dW2 = np.dot(dZ2, self.A1.T) / m  # Gradient for W2
            db2 = np.sum(dZ2) / m  # Gradient for b2
            
            dZ1 = np.dot(self.W2.T, dZ2) * self.A1 * (1 - self.A1)  # Derivative for hidden layer
            dW1 = np.dot(dZ1, X.T) / m  # Gradient for W1
            db1 = np.sum(dZ1) / m  # Gradient for b1
            
            # Update parameters using gradient descent
            self.W1 -= alpha * dW1
            self.b1 -= alpha * db1
            self.W2 -= alpha * dW2
            self.b2 -= alpha * db2
            
            # Store cost
            costs[i] = self.cost(Y, self.A2)
            
            # Print cost every 'step' iterations if verbose is True
            if verbose and i % step == 0:
                print(f"Cost after {i} iterations: {costs[i]}")
        
        return costs  # Return the cost history after training
