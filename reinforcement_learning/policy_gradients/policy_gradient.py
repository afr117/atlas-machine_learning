#!/usr/bin/env python3
"""
Policy Gradient utilities.

Task 0: policy(matrix, weight) -> softmax(matrix @ weight)
Task 1: policy_gradient(state, weight) -> (sampled action, grad of log pi w.r.t. weight)
"""

import numpy as np


def policy(matrix, weight):
    """
    Computes the policy (action probabilities) given state features and weights.

    Args:
        matrix (np.ndarray): shape (n, d), batch of state feature vectors.
        weight (np.ndarray): shape (d, k), weights mapping features -> action logits.

    Returns:
        np.ndarray: shape (n, k), softmax probabilities over actions for each state.
    """
    logits = matrix @ weight
    logits = logits - np.max(logits, axis=-1, keepdims=True)  # numeric stability
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient (∇ log π(a|s) w.r.t. weight) for one state.

    Steps:
      1) π = softmax(state @ weight)
      2) Sample action a ~ π
      3) ∇_W log π(a|s) = s[:, None] * (one_hot(a) - π)

    Args:
        state (np.ndarray): shape (d,) or (1, d), current observation/features.
        weight (np.ndarray): shape (d, k), policy weights.

    Returns:
        action (int): the sampled action.
        grad (np.ndarray): shape (d, k), gradient of log π(a|s) w.r.t. weight.
    """
    s = state.reshape(1, -1)                # (1, d)
    probs = policy(s, weight).reshape(-1)   # (k,)
    k = probs.shape[0]

    # Sample action according to the policy
    action = int(np.random.choice(k, p=probs))

    # One-hot vector for the chosen action
    one_hot = np.zeros_like(probs)
    one_hot[action] = 1.0

    # Gradient: outer product of state (d,) and (one_hot - probs) (k,)
    grad = np.outer(s.reshape(-1), (one_hot - probs))

    return action, grad
