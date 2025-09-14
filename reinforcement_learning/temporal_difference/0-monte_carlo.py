#!/usr/bin/env python3
"""
Monte Carlo state-value prediction (incremental, first-visit).

Works with Gymnasium (0.29.1) discrete environments like FrozenLake.
"""

from typing import Callable, Tuple
import numpy as np


def monte_carlo(env,
                V: np.ndarray,
                policy: Callable[[int], int],
                episodes: int = 5000,
                max_steps: int = 100,
                alpha: float = 0.1,
                gamma: float = 0.99) -> np.ndarray:
    """
    Performs Monte Carlo prediction to update the state-value function V.

    Args:
        env: Gymnasium-like environment instance with discrete observations.
        V (np.ndarray): shape (s,), current value estimates for each state.
        policy (Callable[[int], int]): function mapping state -> action.
        episodes (int): number of episodes to sample.
        max_steps (int): cap on steps per episode.
        alpha (float): step-size (learning rate) for incremental MC updates.
        gamma (float): discount factor in [0, 1].

    Returns:
        np.ndarray: Updated value estimates V with shape (s,).
    """
    for _ in range(episodes):
        # ---- Generate an episode ----
        states: list[int] = []
        rewards: list[float] = []

        obs, _ = env.reset()
        s = int(obs)

        for _t in range(max_steps):
            states.append(s)
            a = policy(s)
            obs, r, terminated, truncated, _info = env.step(a)
            rewards.append(float(r))
            s = int(obs)
            if terminated or truncated:
                break

        # ---- First-visit MC return + incremental update ----
        G = 0.0
        seen: set[int] = set()
        # Walk backward so G is the return from t onward
        for t in range(len(states) - 1, -1, -1):
            G = rewards[t] + gamma * G
            st = states[t]
            # First-visit within the episode
            if st in seen:
                continue
            seen.add(st)
            # Do not update terminal states (keeps holes/goals fixed if pre-labeled)
            # Note: terminal state itself is not in `states` here because we append
            # state BEFORE the step that may terminate. This naturally skips terminal.
            V[st] += alpha * (G - V[st])

    return V

