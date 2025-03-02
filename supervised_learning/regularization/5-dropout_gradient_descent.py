#!/usr/bin/env python3
"""
Updates the weights of a neural network with Dropout regularization using gradient descent.
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization using gradient descent.

    Args:
        Y (numpy.ndarray): One-hot matrix of shape (classes, m) with correct labels.
        weights (dict): Dictionary containing the weights and biases of the network.
        cache (dict): Dictionary containing the outputs and dropout masks of each layer.
        alpha (float): Learning rate.
        keep_prob (float): Probability that a node will be kept.
        L (int): Number of layers of the network.

    Returns:
        None: Updates weights and biases in place.
    """
    m = Y.shape[1]
    dZ = cache[f'A{L}'] - Y  # Gradient of softmax loss

    for i in range(L, 0, -1):
        A_prev = cache[f'A{i-1}']
        W = weights[f'W{i}']
        b = weights[f'b{i}']

        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights[f'W{i}'] -= alpha * dW
        weights[f'b{i}'] -= alpha * db

        if i > 1:
            dZ = np.matmul(W.T, dZ) * (1 - np.power(cache[f'A{i-1}'], 2))
            dZ *= cache[f'D{i-1}']  # Apply dropout mask
            dZ /= keep_prob
