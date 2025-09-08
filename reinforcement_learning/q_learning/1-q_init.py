#!/usr/bin/env python3
"""
Module to initialize the Q-table for a given environment.
"""

import numpy as np


def q_init(env):
    """
    Initializes the Q-table for the given FrozenLake environment.

    Args:
        env: The FrozenLakeEnv instance.

    Returns:
        numpy.ndarray: A Q-table filled with zeros,
        with shape (number of states, number of actions).
    """
    # Number of states in the environment
    n_states = env.observation_space.n
    # Number of possible actions
    n_actions = env.action_space.n

    # Initialize Q-table with zeros
    Q = np.zeros((n_states, n_actions))

    return Q
