#!/usr/bin/env python3
"""
Monte Carlo state-value prediction (first-visit) for discrete environments.
- Only depends on: numpy as np
- Compatible with Gymnasium (0.29.1) FrozenLake-style APIs.
"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.9):
    """
    Performs incremental first-visit Monte Carlo prediction to update V.

    Args:
        env: environment instance with reset() and step() matching Gymnasium.
        V (np.ndarray): shape (s,), value estimates for each state.
        policy (callable): maps state (int) -> action (int).
        episodes (int): number of episodes to sample.
        max_steps (int): cap on steps per episode.
        alpha (float): learning rate for incremental MC updates.
        gamma (float): discount factor in [0, 1].

    Returns:
        np.ndarray: the updated value estimates V (shape (s,)).
    """
    for _ in range(episodes):
        # ---- Generate one episode ----
        states = []
        rewards = []
        obs, _ = env.reset()
        s = int(obs)

        for _t in range(max_steps):
            states.append(s)
            a = policy(s)
            obs, r, terminated, truncated, _ = env.step(a)
            rewards.append(float(r))
            s = int(obs)
            if terminated or truncated:
                break

        # ---- First-visit returns and incremental update ----
        G = 0.0
        visited = set()
        for t in range(len(states) - 1, -1, -1):
            G = rewards[t] + gamma * G
            st = states[t]
            if st in visited:
                continue
            visited.add(st)
            V[st] += alpha * (G - V[st])

    return V

