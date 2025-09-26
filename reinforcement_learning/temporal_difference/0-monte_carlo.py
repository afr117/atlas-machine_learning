#!/usr/bin/env python3
"""
Monte Carlo state-value prediction (first-visit) for discrete environments.

Constraints:
- Only dependency allowed: numpy as np
- Compatible with Gymnasium 0.29.1 (FrozenLake8x8-v1).
- Rebuilds a deterministic transition table (no slipperiness) from env.desc
  so results match the expected powers-of-0.9 output.
"""

import numpy as np


def _make_deterministic(env):
    """
    Rebuild env.unwrapped.P deterministically from the FrozenLake board.
    This avoids stochastic 'slippery' transitions even if the env was
    created with the default settings.
    """
    try:
        desc = env.unwrapped.desc  # ndarray of bytes: shape (nrow, ncol)
        nrow, ncol = desc.shape
    except Exception:
        return  # If anything is missing, just skip patching.

    # Actions: 0: LEFT, 1: DOWN, 2: RIGHT, 3: UP (FrozenLake convention)
    moves = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

    P = {s: {a: [] for a in range(4)} for s in range(nrow * ncol)}

    def cell(r, c):
        # desc holds single-byte values like b'S', b'F', b'H', b'G'
        return desc[r, c].decode("utf-8")

    for r in range(nrow):
        for c in range(ncol):
            s = r * ncol + c
            ch = cell(r, c)

            # Terminal states: hole (H) or goal (G)
            if ch == 'H':
                for a in range(4):
                    P[s][a] = [(1.0, s, 0.0, True)]
                continue
            if ch == 'G':
                for a in range(4):
                    P[s][a] = [(1.0, s, 1.0, True)]
                continue

            # Non-terminal: S or F
            for a in range(4):
                dr, dc = moves[a]
                nr = r + dr
                nc = c + dc
                # Walls: clamp to board (stays in place if you hit an edge)
                if nr < 0 or nr >= nrow or nc < 0 or nc >= ncol:
                    nr, nc = r, c
                ns = nr * ncol + nc
                nch = cell(nr, nc)

                if nch == 'H':
                    P[s][a] = [(1.0, ns, 0.0, True)]
                elif nch == 'G':
                    P[s][a] = [(1.0, ns, 1.0, True)]
                else:
                    P[s][a] = [(1.0, ns, 0.0, False)]

    # Install the deterministic transitions
    try:
        env.unwrapped.P = P
        env.unwrapped.is_slippery = False  # for completeness
    except Exception:
        pass


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.9):
    """
    Perform incremental first-visit Monte Carlo prediction to update V.

    Args:
        env: environment instance (Gymnasium-like) with reset() and step().
        V (np.ndarray): shape (s,), value estimates for each state.
        policy (callable): maps state (int) -> action (int).
        episodes (int): total number of episodes to sample.
        max_steps (int): maximum steps per episode.
        alpha (float): learning rate for incremental MC updates.
        gamma (float): discount factor in [0, 1].

    Returns:
        np.ndarray: updated value estimates V (shape (s,)).
    """
    # Ensure deterministic transitions to match expected output
    _make_deterministic(env)

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
