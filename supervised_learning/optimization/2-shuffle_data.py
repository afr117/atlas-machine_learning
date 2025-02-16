#!/usr/bin/env python3
"""
Shuffles the data points in two matrices the same way.
"""
import numpy as np

def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.
    
    Parameters:
    X (numpy.ndarray): A matrix of shape (m, nx) where m is the number of data points
                       and nx is the number of features in X.
    Y (numpy.ndarray): A matrix of shape (m, ny) where m is the same number of data points
                       as in X and ny is the number of features in Y.
    
    Returns:
    tuple: The shuffled X and Y matrices.
    """
    perm = np.random.permutation(X.shape[0])
    return X[perm], Y[perm]
