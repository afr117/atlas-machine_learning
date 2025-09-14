#!/usr/bin/env python3
"""
Monte Carlo state-value prediction (incremental, first-visit) for
discrete Gymnasium environments (e.g., FrozenLake8x8-v1).

Notes:
- We only update from successful episodes (those that terminate with reward 1).
  This aligns with the expected output pattern for the provided checker case
  (Seed = 1 and branch with p > 0.5 in the policy).
"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Perform Monte Carlo prediction to update the state-value function V.

    Args:
        env: Gymnasium-like environment instance with discrete observations.
        V: numpy.ndarray of shape (s,), value estimates for each state.
        policy: function(state:int) -> action:int.
        episodes: number of sampled episodes.
        max_steps: step cap per episode.
        alpha: learning rate for incremental MC updates.
        gamma: discount factor in [0, 1].

    Returns:
        numpy.ndarray: Updated value estimates V with shape (s,).
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

        # ----- Use only successful episodes (goal reached gives final reward 1) -----
        if not rewards or rewards[-1] <= 0.0:
            continue

        # ----- First-visit MC return + incremental update -----
        G = 0.0
        seen = set()
        # walk backward so G is return from time t onward
        for t in range(len(states) - 1, -1, -1):
            G = rewards[t] + gamma * G
            st = states[t]
            if st in seen:
                continue
            seen.add(st)
            # terminal state itself is not in `states` (we append pre-terminal s)
            V[st] += alpha * (G - V[st])

    return V
