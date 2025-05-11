#!/usr/bin/env python3
"""
This module performs PCA to reduce the dataset to a specified number
of dimensions using eigendecomposition.
"""

import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset to reduce it to `ndim` dimensions.

    Args:
        X (np.ndarray): shape (n, d), original dataset
        ndim (int): target number of dimensions

    Returns:
        np.ndarray: shape (n, ndim), transformed dataset
    """
    # Center the data (mean = 0)
    X_mean = X - np.mean(X, axis=0)

    # Covariance matrix
    cov = np.matmul(X_mean.T, X_mean) / (X.shape[0] - 1)

    # Eigen decomposition
    eig_vals, eig_vecs = np.linalg.eigh(cov)

    # Sort eigenvectors by descending eigenvalues
    idx = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, idx]

    # Select top `ndim` eigenvectors
    W = eig_vecs[:, :ndim]

    # Project data
    T = np.matmul(X_mean, W)

    return T
