#!/usr/bin/env python3
"""
Monte Carlo state-value prediction (first-visit) for discrete environments.

Only dependency: numpy as np.
Compatible with Gymnasium 0.29.1 (reset -> (obs, info), step -> (obs, r, term, trunc, info)).
"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Perform incremental first-visit Monte Carlo prediction to update V.

    Args:
        env: environment instance with reset() and step() like Gymnasium.
        V (np.ndarray): shape (s,), current value estimates for each state.
        policy (callable): maps state (int) -> action (int).
        episodes (int): total number of episodes to sample.
        max_steps (int): maximum steps per episode.
        alpha (float): learning rate for incremental MC.
        gamma (float): discount factor in [0, 1].

    Returns:
        np.ndarray: the updated value estimates V (shape (s,)).
    """
    for _ in range(episodes):
        # ----- Generate one episode -----
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

        # ----- First-visit returns with incremental update -----
        G = 0.0
        seen = set()
        for t in range(len(states) - 1, -1, -1):
            G = rewards[t] + gamma * G
            st = states[t]
            if st in seen:
                continue
            seen.add(st)
            V[st] += alpha * (G - V[st])

    return V
