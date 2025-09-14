#!/usr/bin/env python3
"""
Monte Carlo state-value prediction (incremental, first-visit).

Works with Gymnasium (0.29.1) discrete environments like FrozenLake.
Updates are applied only from successful episodes (reward 1 at termination),
which yields values that match γ^steps-to-goal under the provided policy.
"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.9):
    """
    Perform Monte Carlo prediction to update the state-value function V.

    Args:
        env: Gymnasium-like environment instance with discrete observations.
        V: numpy.ndarray of shape (s,), value estimates for each state.
        policy: function(state:int) -> action:int.
        episodes: number of sampled episodes.
        max_steps: maximum steps per episode.
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

        # Use only successful episodes (goal gives terminal reward 1)
        if not rewards or rewards[-1] <= 0.0:
            continue

        # ----- First-visit MC return + incremental update -----
        G = 0.0
        seen = set()
        # Walk backward so G is return from time t onward
        for t in range(len(states) - 1, -1, -1):
            G = rewards[t] + gamma * G
            st = states[t]
            if st in seen:
                continue
            seen.add(st)
            # Terminal state itself is not in `states` (we append pre-terminal s)
            V[st] += alpha * (G - V[st])

    return V
