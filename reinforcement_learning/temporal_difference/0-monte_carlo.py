#!/usr/bin/env python3
"""
Monte Carlo state-value prediction (incremental, first-visit).

Works with Gymnasium (0.29.1) discrete environments like FrozenLake.
"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Perform Monte Carlo prediction to update the state-value function V.

    Args:
        env: Gymnasium-like environment instance with discrete observations.
        V: numpy.ndarray of shape (s,), current value estimates for each state.
        policy: function mapping state -> action (int).
        episodes: number of episodes to sample.
        max_steps: maximum steps per episode.
        alpha: step-size (learning rate) for incremental MC updates.
        gamma: discount factor in [0, 1].

    Returns:
        numpy.ndarray: Updated value estimates V with shape (s,).
    """
    for _ in range(episodes):
        # ---- Generate an episode ----
        states = []
        rewards = []

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
        seen = set()
        # Walk backward so G is the return from time t onward
        for t in range(len(states) - 1, -1, -1):
            G = rewards[t] + gamma * G
            st = states[t]
            # First-visit within the episode
            if st in seen:
                continue
            seen.add(st)
            # Terminal next-states are not in `states` (we append pre-terminal s)
            V[st] += alpha * (G - V[st])

    return V
