#!/usr/bin/env python3
"""
TD(lambda) state-value prediction with accumulating eligibility traces (on-policy).

Works with Gymnasium (0.29.1) discrete environments like FrozenLake.
"""

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Performs the TD(lambda) algorithm to update the state-value function V.

    Args:
        env: Gymnasium-like environment instance.
        V: np.ndarray of shape (s,), current value estimates.
        policy: function mapping state (int) -> action (int).
        lambtha: eligibility trace decay parameter in [0, 1].
        episodes: number of episodes to run.
        max_steps: cap on steps per episode.
        alpha: learning rate.
        gamma: discount factor.

    Returns:
        np.ndarray: updated value estimates V with shape (s,).
    """
    n_states = V.shape[0]

    for _ in range(episodes):
        # Reset eligibility traces each episode
        E = np.zeros(n_states, dtype=float)

        obs, _ = env.reset()
        s = int(obs)

        for _t in range(max_steps):
            a = policy(s)
            obs_next, r, terminated, truncated, _ = env.step(a)
            s_next = int(obs_next)

            # TD error δ_t
            v_next = 0.0 if (terminated or truncated) else V[s_next]
            delta = r + gamma * v_next - V[s]

            # Accumulating traces: increment current state's trace,
            # then update V and decay all traces.
            E[s] += 1.0
            V += alpha * delta * E
            E *= gamma * lambtha

            s = s_next
            if terminated or truncated:
                break

    return V
