#!/usr/bin/env python3
"""
Performs the Viterbi algorithm for a Hidden Markov Model
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states for a HMM

    Parameters:
    - Observation: np.ndarray of shape (T,) with indices of observations
    - Emission: np.ndarray of shape (N, M), emission probabilities
    - Transition: np.ndarray of shape (N, N), transition probabilities
    - Initial: np.ndarray of shape (N, 1), initial state probabilities

    Returns:
    - path: list of length T with most likely sequence of hidden states
    - P: probability of obtaining the path sequence
    """
    if (not isinstance(Observation, np.ndarray) or Observation.ndim != 1 or
            not isinstance(Emission, np.ndarray) or Emission.ndim != 2 or
            not isinstance(Transition, np.ndarray) or Transition.ndim != 2 or
            not isinstance(Initial, np.ndarray) or Initial.shape[1] != 1 or
            Transition.shape[0] != Transition.shape[1] or
            Emission.shape[0] != Transition.shape[0] or
            Initial.shape[0] != Transition.shape[0]):
        return None, None

    N, T = Emission.shape[0], Observation.shape[0]
    V = np.zeros((N, T))
    backpointer = np.zeros((N, T), dtype=int)

    V[:, 0] = (Initial.T * Emission[:, Observation[0]]).flatten()

    for t in range(1, T):
        for j in range(N):
            prob = V[:, t - 1] * Transition[:, j] * Emission[j, Observation[t]]
            V[j, t] = np.max(prob)
            backpointer[j, t] = np.argmax(V[:, t - 1] * Transition[:, j])

    P = np.max(V[:, -1])
    path = [np.argmax(V[:, -1])]

    for t in range(T - 1, 0, -1):
        path.insert(0, backpointer[path[0], t])

    return path, P
