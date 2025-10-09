#!/usr/bin/env python3
"""Gaussian Process for 1D noiseless case using an RBF kernel."""

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
            X_init (np.ndarray): Inputs sampled from the black-box function,
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
        Calculate the covariance kernel matrix using an RBF kernel.

        Args:
            X1 (np.ndarray): First input matrix of shape (m, 1).
            X2 (np.ndarray): Second input matrix of shape (n, 1).

        Returns:
            np.ndarray: Covariance kernel matrix of shape (m, n).
        """
        a = np.sum(X1 ** 2, axis=1)[:, None]
        b = np.sum(X2 ** 2, axis=1)[None, :]
        sqdist = a + b - 2.0 * (X1 @ X2.T)
        return (self.sigma_f ** 2) * np.exp(-0.5 * sqdist / (self.l ** 2))

    def predict(self, X_s):
        """
        Predict the mean and variance at points X_s.

        Args:
            X_s (np.ndarray): Points of shape (s, 1) to predict.

        Returns:
            tuple:
                mu (np.ndarray): Mean for each point in X_s; shape (s,).
                sigma (np.ndarray): Variance for each point; shape (s,).
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)

        mu = (K_s.T @ self.K_inv @ self.Y).reshape(-1)
        cov = K_ss - (K_s.T @ self.K_inv @ K_s)
        sigma = np.diag(cov)
        return mu, sigma

    def update(self, X_new, Y_new):
        """
        Update the GP with a new sample point.

        Args:
            X_new (np.ndarray): New input sample; shape (1,) or (1, 1).
            Y_new (np.ndarray): New output value; shape (1,) or (1, 1).
        """
        self.X = np.vstack((self.X, X_new.reshape(-1, 1)))
        self.Y = np.vstack((self.Y, Y_new.reshape(-1, 1)))
        self.K = self.kernel(self.X, self.X)
        self.K_inv = np.linalg.inv(self.K)
