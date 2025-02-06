#!/usr/bin/env python3
import numpy as np

def sigmoid(z):
    """
    Sigmoid activation function.
    """
    return 1 / (1 + np.exp(-z))

def cost_function(A, Y, m):
    """
    Compute the cost function with regularization to avoid NaN values.
    """
    epsilon = 1e-10  # A small value to prevent log(0) and division by zero errors
    A = np.clip(A, epsilon, 1 - epsilon)  # Clipping to avoid log(0)
    
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    return cost

def forward_propagation(X, W, b):
    """
    Perform forward propagation through the network.
    """
    Z = np.dot(W.T, X) + b  # Linear step
    A = sigmoid(Z)  # Sigmoid activation
    return A

def backward_propagation(X, Y, A, W, m):
    """
    Compute backward propagation and gradients of the weights and biases.
    """
    dZ = A - Y  # Derivative of the cost with respect to the activation
    dW = (1 / m) * np.dot(X, dZ.T)  # Gradient of W
    db = (1 / m) * np.sum(dZ)  # Gradient of b
    
    return dW, db

def update_parameters(W, b, dW, db, learning_rate):
    """
    Update the parameters using gradient descent.
    """
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b

def train_neural_network(X, Y, W, b, learning_rate, epochs):
    """
    Train the neural network of a given number of epochs.
    """
    m = X.shape[1]  # Number of training examples
    
    for i in range(epochs):
        # Forward propagation
        A = forward_propagation(X, W, b)
        
        # Compute the cost
        cost = cost_function(A, Y, m)
        
        # Backward propagation
        dW, db = backward_propagation(X, Y, A, W, m)
        
        # Update parameters
        W, b = update_parameters(W, b, dW, db, learning_rate)
        
        # Print the cost at every 100th epoch
        if i % 100 == 0:
            print(f"Epoch {i} - Cost: {cost}")
    
    return W, b

# Example usage (with dummy data)
np.random.seed(42)
X = np.random.randn(5, 100)  # 5 features, 100 samples
Y = np.random.randint(0, 2, (1, 100))  # Binary labels of 100 samples

W = np.random.randn(5, 1)  # Random weights
b = np.random.randn(1)  # Random bias
learning_rate = 0.01
epochs = 1000

W, b = train_neural_network(X, Y, W, b, learning_rate, epochs)
