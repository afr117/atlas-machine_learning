#!/usr/bin/env python3
"""
Creates mini-batches for mini-batch gradient descent.
"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches for training using mini-batch gradient descent.

    Parameters:
    X (numpy.ndarray): A matrix of shape (m, nx) where
    m is the number of data points
                       and nx is the number of features.
    Y (numpy.ndarray): A matrix of shape (m, ny)
    where m is the same number of data points
                       as in X and ny is the number of classes
                       for classification tasks.
    batch_size (int): The number of data points per batch.

    Returns:
    list: A list of tuples (X_batch, Y_batch)
    representing the mini-batches.
    """
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    m = X.shape[0]
    mini_batches = []

    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        Y_batch = Y_shuffled[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
