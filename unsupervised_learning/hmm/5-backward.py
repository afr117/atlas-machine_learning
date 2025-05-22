#!/usr/bin/env python3
"""Performs the backward algorithm for a hidden markov model"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model

    Parameters:
    - Observation: np.ndarray of shape (T,) with observation indices
    - Emission: np.ndarray of shape (N, M) with emission probabilities
    - Transition: np.ndarray of shape (N, N) with transition probabilities
    - Initial: np.ndarray of shape (N, 1) with initial state probabilities

    Returns:
    - P: float, likelihood of the observations given the model
    - B: np.ndarray of shape (N, T) with backward path probabilities
    """
    if (not isinstance(Observation, np.ndarray) or Observation.ndim != 1 or
            not isinstance(Emission, np.ndarray) or Emission.ndim != 2 or
            not isinstance(Transition, np.ndarray) or Transition.ndim != 2 or
            not isinstance(Initial, np.ndarray) or Initial.ndim != 2 or
            Emission.shape[0] != Transition.shape[0] or
            Transition.shape[0] != Transition.shape[1] or
            Emission.shape[0] != Initial.shape[0]):
        return None, None

    T = Observation.shape[0]
    N = Emission.shape[0]
    B = np.zeros((N, T))
    B[:, T - 1] = 1

    for t in range(T - 2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(
                B[:, t + 1] * Transition[i, :] * Emission[:, Observation[t + 1]]
            )

    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])
    return P, B
