#!/usr/bin/env python3
import numpy as np

class Neuron:
    def __init__(self, nx):
        """Initializes the neuron"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx <= 0:
            raise ValueError("nx must be a positive integer")
        
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0
    
    @property
    def W(self):
        """Return weights"""
        return self.__W
    
    @property
    def b(self):
        """Return bias"""
        return self.__b
    
    @property
    def A(self):
        """Return activation"""
        return self.__A
    
    def sigmoid(self, Z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):
        """Perform forward propagation to calculate activation"""
        Z = np.dot(self.__W, X) + self.__b
        self.__A = self.sigmoid(Z)
        return self.__A
    
    def cost(self, Y, A):
        """Calculate cost using binary cross-entropy"""
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        return cost
    
    def backward_prop(self, X, Y):
        """Perform backward propagation to calculate gradients"""
        m = X.shape[1]
        dZ = self.__A - Y
        dW = 1 / m * np.dot(dZ, X.T)
        db = 1 / m * np.sum(dZ)
        return dW, db
    
    def update_params(self, dW, db, alpha):
        """Update weights and bias using gradient descent"""
        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Train the neuron"""
        # Validate input
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        
        # Training loop
        for i in range(iterations):
            # Forward propagation
            A = self.forward_prop(X)
            
            # Compute cost
            cost = self.cost(Y, A)
            
            # Backward propagation
            dW, db = self.backward_prop(X, Y)
            
            # Update parameters
            self.update_params(dW, db, alpha)
        
        return self.__A, cost
    
    def evaluate(self, X, Y):
        """Evaluate the neuron performance on data"""
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.round(A)
        return predictions, cost
