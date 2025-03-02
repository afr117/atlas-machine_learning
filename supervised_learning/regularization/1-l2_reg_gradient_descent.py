#!/usr/bin/env python3
"""
Updates the weights and biases of a neural network using
gradient descent with L2 regularization.
"""
import numpy as np

def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates weights and biases using gradient descent with L2 regularization.
    
    Args:
        Y (numpy.ndarray): One-hot labels with shape (classes, m).
        weights (dict): Dictionary of weights and biases of the network.
        cache (dict): Dictionary of network outputs at each layer.
        alpha (float): Learning rate.
        lambtha (float): L2 regularization parameter.
        L (int): Number of layers in the network.
    
    Returns:
        None: Updates weights and biases in place.
    """
    m = Y.shape[1]
    dZ = cache[f'A{L}'] - Y  # Derivative of loss w.r.t. final activation
    
    for i in range(L, 0, -1):
        A_prev = cache[f'A{i-1}']
        W = weights[f'W{i}']
        b = weights[f'b{i}']
        
        dW = (np.matmul(dZ, A_prev.T) / m) + ((lambtha / m) * W.astype(np.float64))
        db = np.sum(dZ, axis=1, keepdims=True) / m
        
        weights[f'W{i}'] -= alpha * dW.astype(np.float64)
        weights[f'b{i}'] -= alpha * db.astype(np.float64)
        
        if i > 1:
            dZ = (1 - np.power(cache[f'A{i-1}'], 2)) * np.matmul(W.T, dZ)
