#!/usr/bin/env python3
"""
Normalizes an unactivated output of a neural network using batch normalization.
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a
    neural network using batch normalization.

    Parameters:
    Z (numpy.ndarray): A matrix of shape (m, n)
    that should be normalized.
    gamma (numpy.ndarray): A matrix of shape (1, n)
    containing the scales used for batch normalization.
    beta (numpy.ndarray): A matrix of shape (1, n)
    containing the offsets used for batch normalization.
    epsilon (float): A small number used to
    avoid division by zero.

    Returns:
    numpy.ndarray: The normalized Z matrix.
    """
    mean = np.mean(Z, axis=0, keepdims=True)
    variance = np.var(Z, axis=0, keepdims=True)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    return gamma * Z_norm + beta
