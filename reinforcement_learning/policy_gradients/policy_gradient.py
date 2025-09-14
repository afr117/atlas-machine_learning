#!/usr/bin/env python3
"""
Policy Gradient utilities.

Implements a simple policy function: softmax(matrix @ weight).
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
    logits = matrix @ weight                      # (n, k)
    logits = logits - np.max(logits, axis=-1, keepdims=True)  # numeric stability
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

