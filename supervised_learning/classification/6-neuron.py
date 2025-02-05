#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Neuron = __import__('6-neuron').Neuron

lib_train = np.load('data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X_train.shape[0])
A, cost = neuron.train(X_train, Y_train, iterations=10)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", np.round(cost, decimals=10))
print("Train accuracy: {}%".format(np.round(accuracy, decimals=10)))
print("Train data:", np.round(A, decimals=10))
print("Train Neuron A:", np.round(neuron.A, decimals=10))

A, cost = neuron.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", np.round(cost, decimals=10))
print("Dev accuracy: {}%".format(np.round(accuracy, decimals=10)))
print("Dev data:", np.round(A, decimals=10))
print("Dev Neuron A:", np.round(neuron.A, decimals=10))

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()                                                                                                                                        and 6-neuron.py: #!/usr/bin/env python3
"""
Neuron class for binary classification

This module defines a Neuron class that implements a single neuron 
performing binary classification, including training and evaluation.
"""

import numpy as np


class Neuron:
    """
    Class that defines a single neuron for binary classification.
    """

    def __init__(self, nx):
        """
        Initializes the Neuron instance.

        Args:
            nx (int): The number of input features.

        Attributes:
            __W (numpy.ndarray): Weights vector of shape (1, nx).
            __b (float): Bias, initialized to 0.
            __A (float): Activation output, initialized to 0.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for the private __W attribute."""
        return self.__W

    @property
    def b(self):
        """Getter for the private __b attribute."""
        return self.__b

    @property
    def A(self):
        """Getter for the private __A attribute."""
        return self.__A

    def sigmoid(self, Z):
        """
        Sigmoid activation function.

        Args:
            Z (numpy.ndarray): Input to the sigmoid function.

        Returns:
            numpy.ndarray: Result after applying sigmoid.
        """
        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):
        """
        Performs forward propagation to calculate the neuron output.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).

        Returns:
            numpy.ndarray: Activation output after applying sigmoid.
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = self.sigmoid(Z)
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost using binary cross-entropy.

        Args:
            Y (numpy.ndarray): True labels of shape (1, m).
            A (numpy.ndarray): Predicted outputs of shape (1, m).

        Returns:
            float: The cost value.
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s performance.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).
            Y (numpy.ndarray): True labels of shape (1, m).

        Returns:
            tuple: The predicted labels (A) and the cost.
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.round(A)
        return predictions, cost

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron using gradient descent.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).
            Y (numpy.ndarray): True labels of shape (1, m).
            iterations (int): Number of iterations to train over.
            alpha (float): Learning rate.

        Returns:
            tuple: The predicted labels after training (A) and the final cost.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        m = X.shape[1]
        for i in range(iterations):
            A = self.forward_prop(X)
            dZ = A - Y
            dW = np.matmul(dZ, X.T) / m
            db = np.sum(dZ) / m
            self.__W -= alpha * dW
            self.__b -= alpha * db

        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        return A, cost                                                                                                                      
