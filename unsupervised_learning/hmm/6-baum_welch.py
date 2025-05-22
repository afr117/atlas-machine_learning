#!/usr/bin/env python3
"""Performs the Baum-Welch algorithm for a hidden markov model"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden markov model

    Parameters:
    - Observations: np.ndarray of shape (T,) containing index of observations
    - Transition: np.ndarray of shape (M, M), initialized transition probabilities
    - Emission: np.ndarray of shape (M, N), initialized emission probabilities
    - Initial: np.ndarray of shape (M, 1), initialized starting probabilities
    - iterations: number of times expectation-maximization should be performed

    Returns:
    - Transition: converged transition probabilities
    - Emission: converged emission probabilities
    - or (None, None) on failure
    """
    if (not isinstance(Observations, np.ndarray) or Observations.ndim != 1 or
        not isinstance(Emission, np.ndarray) or Emission.ndim != 2 or
        not isinstance(Transition, np.ndarray) or Transition.ndim != 2 or
        not isinstance(Initial, np.ndarray) or Initial.ndim != 2):
        return None, None

    T = Observations.shape[0]
    M, N = Emission.shape

    for _ in range(iterations):
        alpha = np.zeros((M, T))
        beta = np.zeros((M, T))
        alpha[:, 0] = Initial.T * Emission[:, Observations[0]]

        for t in range(1, T):
            for j in range(M):
                alpha[j, t] = np.sum(alpha[:, t - 1] * Transition[:, j]) * \
                              Emission[j, Observations[t]]

        beta[:, -1] = 1
        for t in range(T - 2, -1, -1):
            for i in range(M):
                beta[i, t] = np.sum(
                    Transition[i, :] * Emission[:, Observations[t + 1]] * beta[:, t + 1]
                )

        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denom = np.dot(
                np.dot(alpha[:, t].T, Transition) *
                Emission[:, Observations[t + 1]].T, beta[:, t + 1]
            )
            for i in range(M):
                numer = alpha[i, t] * Transition[i, :] * \
                        Emission[:, Observations[t + 1]] * beta[:, t + 1]
                xi[i, :, t] = numer / denom

        gamma = np.sum(xi, axis=1)
        prod = alpha * beta
        gamma = np.hstack((gamma, np.sum(prod[:, -1:], axis=0, keepdims=True)))

        Initial = gamma[:, [0]].copy()
        Transition = np.sum(xi, axis=2) / np.sum(gamma[:, :-1], axis=1)[:, None]

        for i in range(M):
            for j in range(N):
                mask = Observations == j
                Emission[i, j] = np.sum(gamma[i, mask]) / np.sum(gamma[i, :])

    return Transition, Emission
