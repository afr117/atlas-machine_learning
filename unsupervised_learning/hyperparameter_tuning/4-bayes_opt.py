#!/usr/bin/env python3
"""Bayesian Optimization module with acquisition function"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process.

    Attributes:
        f (function): The black-box function to optimize.
        gp (GaussianProcess): The underlying Gaussian process model.
        X_s (np.ndarray): Acquisition sample points.
        xsi (float): Exploration-exploitation factor.
        minimize (bool): Whether to minimize or maximize the function.
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Class constructor.

        Args:
            f (function): Black-box function to optimize.
            X_init (np.ndarray): Initial sampled inputs, shape (t, 1).
            Y_init (np.ndarray): Initial sampled outputs, shape (t, 1).
            bounds (tuple): Tuple (min, max) defining the domain bounds.
            ac_samples (int): Number of acquisition sample points.
            l (float): Kernel length parameter.
            sigma_f (float): Kernel output standard deviation.
            xsi (float): Exploration-exploitation factor.
            minimize (bool): True for minimization, False for maximization.
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1],
                               ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location using Expected Improvement.

        Returns:
            X_next (np.ndarray of shape (1,)): Next best sample point.
            EI (np.ndarray of shape (ac_samples,)):
            Expected improvement values.
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
