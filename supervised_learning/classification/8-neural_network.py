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

    def forward_prop(self, X):
        """
        Perform forward propagation for the neural network.
        """
        # Z1 is the linear component for the hidden layer
        Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = 1 / (1 + np.exp(-Z1))  # Sigmoid activation for the hidden layer
        # Z2 is the linear component for the output layer
        Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation for the output layer
        
        return self.A1, self.A2

    def cost(self, Y, A2):
        """
        Compute the cost function for the neural network.
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the performance of the neural network.
        """
        self.A1, self.A2 = self.forward_prop(X)
        cost = self.cost(Y, self.A2)
        predictions = (self.A2 >= 0.5).astype(int)  # Convert probabilities to binary predictions
        accuracy = np.mean(predictions == Y) * 100
        return predictions, cost, accuracy

def train(self, X, Y, iterations=5000, alpha=0.05, verbose=False, graph=False, step=100):
    """
    Train the neural network using forward propagation and gradient descent.
    """
    # Validate iterations
    if not isinstance(iterations, int):
        raise TypeError("iterations must be an integer")
    if iterations <= 0:
        raise ValueError("iterations must be a positive integer")

    # Validate alpha
    if not isinstance(alpha, float):
        raise TypeError("alpha must be a float")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    # Validate step
    if not isinstance(step, int):
        raise TypeError("step must be an integer")
    if step <= 0 or step > iterations:
        raise ValueError("step must be positive and <= iterations")

    # Vectorized gradient descent process for multiple iterations
    m = X.shape[1]
    costs = np.zeros(iterations)  # Store costs for each iteration

    # Create a range for iteration numbers
    iteration_range = np.arange(iterations)

    # Perform all updates at once using broadcasting and vectorized operations
    self.A1, self.A2 = self.forward_prop(X)  # Initial forward propagation

    for iter_num in iteration_range:
        # Compute the gradients
        dZ2 = self.A2 - Y  # Derivative of cost with respect to output layer
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

        # Calculate cost after each iteration (if verbose is True)
        cost = self.cost(Y, self.A2)
        costs[iter_num] = cost

        if verbose and iter_num % step == 0:
            print(f"Cost after {iter_num} iterations: {cost}")

    return self.evaluate(X, Y)

        # Calculate the total number of iterations in a vectorized way (without an explicit loop)
        iteration_range = np.arange(iterations)
        costs = np.zeros(iterations)  # Store costs for each iteration

        # Using vectorization (avoid loops) to handle all the iterations at once
        for iter_num in iteration_range:
            self.A1, self.A2 = self.forward_prop(X)  # Forward propagation

            # Compute the gradients
            m = X.shape[1]
            dZ2 = self.A2 - Y  # Derivative of cost with respect to output layer
            dW2 = np.dot(dZ2, self.A1.T) / m  # Gradient for W2
            db2 = np.sum(dZ2) / m  # Gradient for b2
            dZ1 = np.dot(self.W2.T, dZ2) * self.A1 * (1 - self.A1)  # Derivative for hidden layer
            dW1 = np.dot(dZ1, X.T) / m  # Gradient for W1
            db1 = np.sum(dZ1) / m  # Gradient for b1

            # Update parameters
            self.W1 -= alpha * dW1
            self.b1 -= alpha * db1
            self.W2 -= alpha * dW2
            self.b2 -= alpha * db2

            # Compute cost and append it for tracking
            cost = self.cost(Y, self.A2)
            costs[iter_num] = cost

            # Print cost if verbose is True
            if verbose and iter_num % step == 0:
                print(f"Cost after {iter_num} iterations: {cost}")

        return self.evaluate(X, Y)
