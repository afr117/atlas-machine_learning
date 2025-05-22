#!/usr/bin/env python3
"""Performs the Baum-Welch algorithm for a hidden markov model"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    T = Observation.shape[0]
    N = Emission.shape[0]
    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.sum(F[:, t - 1] * Transition[:, j]) * Emission[j, Observation[t]]

    P = np.sum(F[:, -1])
    return P, F


def backward(Observation, Emission, Transition, Initial):
    T = Observation.shape[0]
    N = Emission.shape[0]
    B = np.zeros((N, T))
    B[:, -1] = 1

    for t in range(T - 2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(Transition[i] * Emission[:, Observation[t + 1]] * B[:, t + 1])

    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])
    return P, B


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    if (not isinstance(Observations, np.ndarray) or
        not isinstance(Transition, np.ndarray) or
        not isinstance(Emission, np.ndarray) or
        not isinstance(Initial, np.ndarray) or
        not isinstance(iterations, int) or iterations <= 0):
        return None, None

    T = Observations.shape[0]
    N, M = Emission.shape

    for _ in range(iterations):
        P, F = forward(Observations, Emission, Transition, Initial)
        _, B = backward(Observations, Emission, Transition, Initial)

        xi = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            denom = np.dot(F[:, t], np.dot(Transition, Emission[:, Observations[t + 1]] * B[:, t + 1]))
            for i in range(N):
                numer = F[i, t] * Transition[i] * Emission[:, Observations[t + 1]] * B[:, t + 1]
                xi[i, :, t] = numer / denom

        gamma = np.sum(xi, axis=1)
        gamma_last = np.sum(F[:, -1] * B[:, -1], keepdims=True)
        gamma = np.hstack((gamma, gamma_last.reshape(-1, 1)))

        Transition = np.sum(xi, axis=2) / np.sum(gamma[:, :-1], axis=1, keepdims=True)

        for i in range(N):
            for j in range(M):
                mask = (Observations == j)
                Emission[i, j] = np.sum(gamma[i, mask]) / np.sum(gamma[i])

    return Transition, Emission
