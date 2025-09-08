#!/usr/bin/env python3
"""
Module that loads the FrozenLake environment from gymnasium
"""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLakeEnv environment from gymnasium.

    Args:
        desc (list[list[str]] | None): Custom map description.
        map_name (str | None): Pre-made map name, e.g. '4x4', '8x8'.
        is_slippery (bool): Whether the ice is slippery.

    Returns:
        gym.Env: The initialized FrozenLake environment.
    """
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode="ansi",          # 👈 required for text rendering
    )
    return env
