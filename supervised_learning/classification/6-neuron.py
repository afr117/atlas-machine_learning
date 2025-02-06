#!/usr/bin/env python3

import numpy as np
class Neuron:
    def __init__(self, nx):
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
        return self.__W
    
    @property
    def b(self):
        return self.__b
    
    @property
    def A(self):
        return self.__A
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def forward_prop(self, X):
        Z = np.dot(self.__W, X) + self.__b
        self.__A = self.sigmoid(Z)
        return self.__A
    
    def cost(self, Y, A):
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        return cost
    
    def backward_prop(self, X, Y):
        m = X.shape[1]
        dZ = self.__A - Y
        dW = 1 / m * np.dot(dZ, X.T)
        db = 1 / m * np.sum(dZ)
        return dW, db
    
    def update_params(self, dW, db, alpha):
        self.__W -= alpha * dW
        self.__b -= alpha * db
    
    def train(self, X, Y, iterations=5000, alpha=0.05):
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        
        # Merging training and evaluation steps into a single loop
        for i in range(iterations):
            # Forward propagation
            A = self.forward_prop(X)
            
            # Compute cost
            cost = self.cost(Y, A)
            
            # Backward propagation
            dW, db = self.backward_prop(X, Y)
            
            # Update parameters
            self.update_params(dW, db, alpha)
            
            # Evaluate performance on each iteration
            if i % 500 == 0 or i == iterations - 1:  # Every 500 iterations or last iteration
                # Training performance
                train_accuracy = np.sum(A == Y) / Y.shape[1] * 100
                print(f"Iteration {i}/{iterations}:")
                print("Train cost:", np.round(cost, decimals=10))
                print(f"Train accuracy: {np.round(train_accuracy, decimals=10)}%")
                
        return self.__A, cost
    
    def evaluate(self, X, Y):
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.round(A)
        accuracy = np.sum(predictions == Y) / Y.shape[1] * 100
        return predictions, cost, accuracy


# Main code to test the neuron
if __name__ == '__main__':
    # Loading training data
    lib_train = np.load('data/Binary_Train.npz')
    X_train_3D, Y_train = lib_train['X'], lib_train['Y']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
    
    # Loading development data
    lib_dev = np.load('data/Binary_Dev.npz')
    X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
    X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T
    
    # Initializing the neuron
    np.random.seed(0)
    neuron = Neuron(X_train.shape[0])
    
    # Training the neuron with 5000 iterations
    A, cost = neuron.train(X_train, Y_train, iterations=5000, alpha=0.05)
    
    # Evaluating on development data after training is complete
    predictions, cost_dev, accuracy_dev = neuron.evaluate(X_dev, Y_dev)
    
    # Print final evaluation results
    print("Final Dev cost:", np.round(cost_dev, decimals=10))
    print(f"Final Dev accuracy: {np.round(accuracy_dev, decimals=10)}%")
    print("Dev predictions:", np.round(predictions, decimals=10))
    print("Final Neuron A:", np.round(neuron.A, decimals=10))
