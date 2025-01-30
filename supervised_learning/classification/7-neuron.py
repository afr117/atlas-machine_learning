#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        # Initialize weights (W) and bias (b) for the neuron
        self.__W = np.random.randn(1, nx)  # Weight matrix initialized to random values
        self.__b = 0  # Bias initialized to 0
        self.__A = 0  # Activated output initialized to 0

    def forward_prop(self, X):
        """
        Implements forward propagation of the neuron.
        X: Input data, numpy array with shape (nx, m)
        Returns the activated output A
        """
        Z = np.dot(self.__W, X) + self.__b  # Linear part of the forward propagation
        self.__A = 1 / (1 + np.exp(-Z))  # Sigmoid activation function
        return self.__A

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Performs one step of gradient descent on the neuron.
        X: Input data, numpy array with shape (nx, m)
        Y: True labels, numpy array with shape (1, m)
        A: Activated output of the neuron, numpy array with shape (1, m)
        alpha: Learning rate (float)
        Updates the weights (W) and bias (b)
        """
        m = X.shape[1]  # Number of examples
        dz = A - Y  # Derivative of the cost with respect to the output
        dw = np.dot(dz, X.T) / m  # Derivative of the cost with respect to W
        db = np.sum(dz) / m  # Derivative of the cost with respect to b

        # Update weights and bias
        self.__W -= alpha * dw
        self.__b -= alpha * db

    def cost(self, Y, A):
        """
        Computes the cost using binary cross-entropy.
        Y: True labels, numpy array with shape (1, m)
        A: Activated output of the neuron, numpy array with shape (1, m)
        Returns the cost (scalar)
        """
        m = Y.shape[1]  # Number of examples
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
        return cost

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the neuron by updating the private attributes __W, __b, and __A.
        X: Input data, numpy array with shape (nx, m)
        Y: True labels, numpy array with shape (1, m)
        iterations: Number of iterations for training (int)
        alpha: Learning rate (float)
        verbose: If True, print cost every step iterations
        graph: If True, plot the training cost graph after training
        step: Interval of iterations to print cost and graph data
        Returns the final activated output and final cost after training
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

        # Initialize variables for cost tracking
        costs = []

        # Training loop
        for i in range(iterations + 1):
            # Perform forward propagation and gradient descent in the same loop
            A = self.forward_prop(X)  # Forward propagation
            self.gradient_descent(X, Y, A, alpha)  # Perform gradient descent

            # Compute the cost every step iterations and save it
            if i % step == 0:
                cost = self.cost(Y, A)
                costs.append(cost)

                # Verbose output: Print the cost after every step iterations
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")

        # Plot the cost graph if required
        if graph:
            plt.plot(range(0, iterations + 1, step), costs, label="Cost")
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.title("Training Cost")
            plt.legend()
            plt.show()

        # Return final activated output and cost after training
        A = self.forward_prop(X)  # Final forward propagation
        cost = self.cost(Y, A)  # Final cost computation
        return A, cost

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def evaluate(self, X, Y):
        """
        Evaluates the neuron on the given data.
        X: Input data, numpy array with shape (nx, m)
        Y: True labels, numpy array with shape (1, m)
        Returns the activated output and the cost
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        return A, cost
