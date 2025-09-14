#!/usr/bin/env python3
"""
TD(λ) state-value prediction with accumulating eligibility traces.

Works with Gymnasium (0.29.1) discrete environments like FrozenLake.
On each step:
  δ = r + γ * V[s_next] (or 0 if terminal) - V[s]
  e[s] += 1
  V += α * δ * e
  e *= γ * λ
"""

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Performs the TD(λ) algorithm to update the state-value function V.

    Args:
        env: Gymnasium-like environment instance with discrete observations.
        V: numpy.ndarray of shape (s,), current value estimates for each state.
        policy: function mapping state (int) -> action (int).
        lambtha: eligibility trace factor λ in [0, 1].
        episodes: total number of episodes to train over.
        max_steps: maximum number of steps per episode.
        alpha: learning rate.
        gamma: discount factor in [0, 1].

    Returns:
        numpy.ndarray: Updated value estimates V with shape (s,).
    """
    n_states = V.shape[0]

    for _ in range(episodes):
        # reset episode
        e = np.zeros(n_states, dtype=float)   # eligibility traces
        obs, _ = env.reset()
        s = int(obs)

        for _t in range(max_steps):
            a = policy(s)
            obs, r, terminated, truncated, _ = env.step(a)
            s_next = int(obs)

            v_next = 0.0 if (terminated or truncated) else V[s_next]
            delta = float(r) + gamma * v_next - V[s]

            # accumulate trace for current state and update all states
            e[s] += 1.0
            V += alpha * delta * e

            # decay traces
            e *= gamma * lambtha

            if terminated or truncated:
                break

            s = s_next

    return V
