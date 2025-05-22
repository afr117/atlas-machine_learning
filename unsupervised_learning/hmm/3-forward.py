#!/usr/bin/env python3
"""
Performs the forward algorithm for a Hidden Markov Model
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Calculates the forward path probabilities of a HMM

    Parameters:
    - Observation: np.ndarray of shape (T,) with the index of observations
    - Emission: np.ndarray of shape (N, M), emission probabilities
    - Transition: np.ndarray of shape (N, N), transition probabilities
    - Initial: np.ndarray of shape (N, 1), initial state probabilities

    Returns:
    - P: likelihood of the observations given the model
    - F: np.ndarray of shape (N, T), forward path probabilities
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
    F = np.zeros((N, T))

    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.sum(F[:, t - 1] * Transition[:, j]) * \
                Emission[j, Observation[t]]

    P = np.sum(F[:, -1])
    return P, F
