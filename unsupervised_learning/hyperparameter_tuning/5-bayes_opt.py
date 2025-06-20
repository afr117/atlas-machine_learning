#!/usr/bin/env python3
"""Bayesian Optimization module with optimization loop"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process.
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Class constructor.
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1],
                               ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
        self.bounds = bounds

    def acquisition(self):
        """
        Calculates the next best sample location using Expected Improvement.

        Returns:
            X_next (np.ndarray of shape (1,)): Next best sample point.
            EI (np.ndarray of shape (ac_samples,)): Expected improvement values.
        """
        mu, sigma = self.gp.predict(self.X_s)
        sigma = np.where(sigma == 0, 1e-10, sigma)

        if self.minimize:
            best = np.min(self.gp.Y)
            imp = best - mu - self.xsi
        else:
            best = np.max(self.gp.Y)
            imp = mu - best - self.xsi

        Z = imp / sigma
        EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        EI = np.where(sigma == 0, 0, EI)

        X_next = self.X_s[np.argmax(EI)].reshape(1,)
        return X_next, EI

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function.

        Args:
            iterations (int): Maximum number of iterations.

        Returns:
            X_opt (np.ndarray of shape (1,)): Optimal input point.
            Y_opt (np.ndarray of shape (1,)): Optimal function value.
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            # Check if X_next already in X
            if np.any(np.isclose(self.gp.X, X_next).all(axis=1)):
                break

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx].reshape(1,)
        Y_opt = self.gp.Y[idx].reshape(1,)

        return X_opt, Y_opt
