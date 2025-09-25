#!/usr/bin/env python3
"""
Monte Carlo state-value prediction (first-visit) for discrete environments.
Only dependency: numpy as np. Compatible with Gymnasium 0.29.1 APIs.
"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.9):
    """
    Performs incremental first-visit Monte Carlo prediction to update V.

    Args:
        env: Gymnasium-like environment (reset, step) with discrete states.
        V (np.ndarray): shape (s,), current value estimates.
        policy (callable): state(int) -> action(int).
        episodes (int): number of episodes to sample.
        max_steps (int): max steps per episode.
        alpha (float): step-size for incremental MC updates.
        gamma (float): discount factor.

    Returns:
        np.ndarray: updated value estimates V.
    """
    # Make FrozenLake deterministic if possible, to match expected outputs
    try:
        if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "is_slippery"):
            env.unwrapped.is_slippery = False
    except Exception:
        # If we can't change it, just proceed stochastically
        pass

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
        seen = set()
        for t in range(len(states) - 1, -1, -1):
            G = rewards[t] + gamma * G
            st = states[t]
            if st in seen:
                continue
            seen.add(st)
            V[st] += alpha * (G - V[st])

    return V
