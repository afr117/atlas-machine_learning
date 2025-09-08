#!/usr/bin/env python3
"""
Module to implement the epsilon-greedy policy.
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Selects the next action using the epsilon-greedy strategy.

    Args:
        Q (numpy.ndarray): The Q-table.
        state (int): The current state index.
        epsilon (float): The probability of exploring.

    Returns:
        int: The chosen action index.
    """
    # Exploration vs exploitation decision
    p = np.random.uniform(0, 1)

    if p < epsilon:
        # Explore: choose a random action
        action = np.random.randint(Q.shape[1])
    else:
        # Exploit: choose the best action from Q-table
        action = np.argmax(Q[state])

    return action
