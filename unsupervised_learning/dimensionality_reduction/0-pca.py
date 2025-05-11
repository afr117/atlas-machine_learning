#!/usr/bin/env python3
"""
This module performs Principal Component Analysis (PCA)
on a centered dataset.
"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset to reduce its dimensionality
    while maintaining the specified amount of variance.

    Args:
        X (np.ndarray): shape (n, d), centered data
        var (float): fraction of variance to preserve (default 0.95)

    Returns:
        W (np.ndarray): shape (d, nd), projection matrix that preserves
                        at least `var` fraction of the original variance
    """
    # Calculate the covariance matrix
    cov = np.matmul(X.T, X) / (X.shape[0] - 1)

    # Perform eigen decomposition
    eig_vals, eig_vecs = np.linalg.eigh(cov)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    # Compute cumulative variance ratio
    total_var = np.sum(eig_vals)
    cum_var = np.cumsum(eig_vals) / total_var

    # Find the number of components to preserve the given variance
    nd = np.searchsorted(cum_var, var) + 1

    # Return the weights matrix W with selected components
    W = eig_vecs[:, :nd]

    return W
