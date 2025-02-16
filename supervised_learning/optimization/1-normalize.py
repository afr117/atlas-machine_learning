#!/usr/bin/env python3
"""
Normalizes (standardizes) a matrix using given mean and standard deviation.
"""
import numpy as np

def normalize(X, m, s):
    """
    Normalizes a matrix.
    
    Parameters:
    X (numpy.ndarray): A matrix of shape (d, nx) where d is the number of data points
                       and nx is the number of features.
    m (numpy.ndarray): A 1D array of shape (nx,) containing the mean of each feature.
    s (numpy.ndarray): A 1D array of shape (nx,) containing the standard deviation of each feature.
    
    Returns:
    numpy.ndarray: The normalized matrix.
    """
    return (X - m) / s
