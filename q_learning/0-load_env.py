#!/usr/bin/env python3
"""
Module that loads the FrozenLake environment from gymnasium
"""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLakeEnv environment from gymnasium.

    Args:
        desc (list of lists of str, optional): Custom description of the map.
            Each list represents a row, with characters like:
            'S' (start), 'F' (frozen), 'H' (hole), 'G' (goal).
            Defaults to None.
        map_name (str, optional): Pre-made map name (e.g., '4x4', '8x8').
            Defaults to None.
        is_slippery (bool, optional): If True, ice is slippery.
            Defaults to False.

    Returns:
        gym.Env: The initialized FrozenLake environment.
    """
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )
    return env
