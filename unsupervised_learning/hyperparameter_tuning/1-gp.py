#!/usr/bin/env python3
"""Gaussian Process module for 1D noiseless case using RBF kernel"""

import numpy as np


class GaussianProcess:
    """
    Represents a noiseless 1D Gaussian process.

    Attributes:
        X (np.ndarray): Sampled inputs.
        Y (np.ndarray): Sampled outputs corresponding to X.
        l (float): Length scale for the RBF kernel.
        sigma_f (float): Standard deviation of the output.
        K (np.ndarray): Covariance kernel matrix of X.
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor.

        Args:
            X_init (np.ndarray): Inputs sampled with the black-box function,
                                 shape (t, 1).
            Y_init (np.ndarray): Outputs from the black-box function,
                                 shape (t, 1).
            l (float): Length parameter for the RBF kernel.
            sigma_f (float): Standard deviation of the output.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)
        self.K_inv = np.linalg.inv(self.K)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix using RBF.

        Args:
            X1 (np.ndarray): First input matrix of shape (m, 1).
            X2 (np.ndarray): Second input matrix of shape (n, 1).

        Returns:
            np.ndarray: Covariance kernel matrix of shape (m, n).
        """
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) \ + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """
        Predicts the mean and variance of points in a Gaussian process.

        Args:
            X_s (np.ndarray): Points of shape (s, 1) to predict.

        Returns:
            tuple: mu, sigma
                - mu (np.ndarray of shape (s,)): Mean for each point in X_s
                - sigma (np.ndarray of shape (s,)): Variance for point in X_s
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        mu = K_s.T.dot(self.K_inv).dot(self.Y).reshape(-1)
        cov = K_ss - K_s.T.dot(self.K_inv).dot(K_s)
        sigma = np.diag(cov)
        return mu, sigma
