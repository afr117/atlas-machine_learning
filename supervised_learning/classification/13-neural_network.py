#!/usr/bin/env python3
import numpy as np

class NeuralNetwork:
    
    def __init__(self, nx, nodes):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        
        # Initialize the weights, biases, and activated outputs of both layers
        self.W1 = np.random.randn(nodes, nx)  # Weights of the hidden layer
        self.b1 = np.zeros((nodes, 1))  # Bias of the hidden layer
        self.A1 = np.zeros((nodes, 1))  # Activated output of the hidden layer
        
        self.W2 = np.random.randn(1, nodes)  # Weights of the output layer
        self.b2 = np.zeros((1, 1))  # Bias of the output layer
        self.A2 = np.zeros((1, 1))  # Activated output of the output layer
    
    def forward_prop(self, X):
        Z1 = np.dot(self.W1, X) + self.b1  # Linear transformation of the hidden layer
        self.A1 = 1 / (1 + np.exp(-Z1))  # Sigmoid activation function of the hidden layer
        
        Z2 = np.dot(self.W2, self.A1) + self.b2  # Linear transformation of the output layer
        self.A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation function of the output layer
        
        return self.A1, self.A2
    
    def cost(self, Y, A):
        m = Y.shape[1]  # Number of examples
        # Compute the binary cross-entropy cost
        cost = -np.mean(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost
    
    def evaluate(self, X, Y):
        _, A = self.forward_prop(X)  # Perform forward propagation
        
        # Predictions: 1 if A >= 0.5, else 0 (vectorized)
        predictions = (A >= 0.5).astype(int)
        
        # Calculate the cost using the cost method
        cost = self.cost(Y, A)
        
        return predictions, cost
    
    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        m = X.shape[1]  # Number of examples
        
        # Compute the derivatives of the cost with respect to W2, b2, W1, b1
        dz2 = A2 - Y  # Derivative of the cost with respect to A2
        dw2 = np.dot(dz2, A1.T) / m  # Derivative of the cost with respect to W2
        db2 = np.sum(dz2) / m  # Derivative of the cost with respect to b2
        
        dz1 = np.dot(self.W2.T, dz2) * A1 * (1 - A1)  # Derivative of the cost with respect to A1
        dw1 = np.dot(dz1, X.T) / m  # Derivative of the cost with respect to W1
        db1 = np.sum(dz1, axis=1, keepdims=True) / m  # Derivative of the cost with respect to b1
        
        # Update the weights and biases using gradient descent (no loops)
        self.W2 -= alpha * dw2
        self.b2 -= alpha * db2
        self.W1 -= alpha * dw1
        self.b1 -= alpha * db1
