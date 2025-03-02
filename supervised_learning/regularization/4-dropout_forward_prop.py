#!/usr/bin/env python3
"""
Conducts forward propagation using Dropout.
"""
import numpy as np

def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.
    
    Args:
        X (numpy.ndarray): Input data for the network of shape (nx, m).
        weights (dict): Dictionary containing the weights and biases of the network.
        L (int): Number of layers in the network.
        keep_prob (float): Probability that a node will be kept.
    
    Returns:
        dict: Dictionary containing the outputs of each layer and dropout masks.
    """
    cache = {'A0': X}
    
    for i in range(1, L + 1):
        W = weights[f'W{i}']
        b = weights[f'b{i}']
        Z = np.matmul(W, cache[f'A{i-1}']) + b
        
        if i == L:
            # Softmax activation for the last layer
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        else:
            # Tanh activation function with dropout
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            A *= D
            A /= keep_prob
            cache[f'D{i}'] = D.astype(int)  # Ensure dropout mask is binary (0/1)
        
        cache[f'A{i}'] = A
    
    return cache
