#!/usr/bin/env python3
import numpy as np

class Neuron:
    def __init__(self, nx):
        """Initializes the neuron"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx <= 0:
            raise ValueError("nx must be a positive integer")
        
        # Random initialization for weights
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
        
        # Training loop with more iterations for proper training
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


# Main code to test
import numpy as np
Neuron = __import__('6-neuron').Neuron

# Loading training data
lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T

# Loading development data
lib_dev = np.load('../data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

# Initializing the neuron
np.random.seed(0)
neuron = Neuron(X_train.shape[0])

# Training the neuron with 5000 iterations
A, cost = neuron.train(X_train, Y_train, iterations=5000, alpha=0.05)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", np.round(cost, decimals=10))
print("Train accuracy: {}%".format(np.round(accuracy, decimals=10)))
print("Train data:", np.round(A, decimals=10))
print("Train Neuron A:", np.round(neuron.A, decimals=10))

# Evaluating the neuron on dev data
A, cost = neuron.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", np.round(cost, decimals=10))
print("Dev accuracy: {}%".format(np.round(accuracy, decimals=10)))
print("Dev data:", np.round(A, decimals=10))
print("Dev Neuron A:", np.round(neuron.A, decimals=10))
