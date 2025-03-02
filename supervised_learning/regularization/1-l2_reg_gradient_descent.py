#!/usr/bin/env python3
"""Module to update weights and biases using gradient descent with L2 regularization."""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent
    with L2 regularization.

    Args:
        Y (numpy.ndarray): One-hot matrix of shape (classes, m) with correct labels.
        weights (dict): Dictionary containing the weights and biases of the network.
        cache (dict): Dictionary containing the outputs of each layer.
        alpha (float): Learning rate.
        lambtha (float): L2 regularization parameter.
        L (int): Number of layers in the network.

    Returns:
        None: Updates the weights and biases in place.
    """
    m = Y.shape[1]
    dZ = cache[f"A{L}"] - Y  # Derivative of cost w.r.t output layer

    for i in range(L, 0, -1):
        A_prev = cache[f"A{i-1}"]
        W = weights[f"W{i}"]

        # Compute gradients with L2 regularization
        dW = (np.matmul(dZ, A_prev.T) / m) + (lambtha / m) * W
        db = np.sum(dZ, axis=1, keepdims=True) / m

        # Update weights and biases
        weights[f"W{i}"] -= alpha * dW
        weights[f"b{i}"] -= alpha * db

        # Compute dZ for next layer (if not input layer)
        if i > 1:
            dZ = np.matmul(W.T, dZ) * (1 - np.square(A_prev))  # tanh derivative
