#!/usr/bin/env python3
"""Performs the backward algorithm for a hidden markov model"""
import numpy as np

def backward(Observation, Emission, Transition, Initial):
    """Calculates the backward path probabilities for a hidden markov model"""
    if (not isinstance(Observation, np.ndarray) or Observation.ndim != 1 or
        not isinstance(Emission, np.ndarray) or Emission.ndim != 2 or
        not isinstance(Transition, np.ndarray) or Transition.ndim != 2 or
        not isinstance(Initial, np.ndarray) or Initial.shape[1] != 1):
        return None, None

    T = Observation.shape[0]
    N = Emission.shape[0]

    if Transition.shape != (N, N) or Initial.shape[0] != N:
        return None, None

    B = np.zeros((N, T))
    B[:, T - 1] = 1

    for t in range(T - 2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(B[:, t + 1] * Transition[i, :] * Emission[:, Observation[t + 1]])

    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])
    return P, B
