#!/usr/bin/env python3
"""Performs the EM algorithm for a Gaussian Mixture Model"""
import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the EM algorithm for a GMM

    Parameters:
    - X: np.ndarray of shape (n, d), data set
    - k: int, number of clusters
    - iterations: int, maximum number of iterations
    - tol: float, tolerance for log likelihood convergence
    - verbose: bool, whether to print log likelihood info

    Returns:
    - pi: np.ndarray of shape (k,), priors
    - m: np.ndarray of shape (k, d), means
    - S: np.ndarray of shape (k, d, d), covariances
    - g: np.ndarray of shape (k, n), responsibilities
    - log_likelihood: final log likelihood
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    g, log_likelihood = expectation(X, pi, m, S)
    if g is None or log_likelihood is None:
        return None, None, None, None, None

    for i in range(iterations):
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        g, new_ll = expectation(X, pi, m, S)
        if g is None or new_ll is None:
            return None, None, None, None, None

        if verbose and (i % 10 == 0 or i == iterations - 1):
            print(f"Log Likelihood after {i} iterations: {log_likelihood:.5f}")

        if abs(new_ll - log_likelihood) <= tol:
            log_likelihood = new_ll
            if verbose:
                print(f"Log Likelihood after {i + 1} iterations: {log_likelihood:.5f}")
            return pi, m, S, g, log_likelihood

        log_likelihood = new_ll

    return pi, m, S, g, log_likelihood
