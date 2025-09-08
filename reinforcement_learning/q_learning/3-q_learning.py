#!/usr/bin/env python3
"""
Q-learning trainer for FrozenLake (Gymnasium).
"""

import numpy as np
from typing import Tuple, List
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env,
          Q: np.ndarray,
          episodes: int = 5000,
          max_steps: int = 100,
          alpha: float = 0.1,
          gamma: float = 0.99,
          epsilon: float = 1.0,
          min_epsilon: float = 0.1,
          epsilon_decay: float = 0.05) -> Tuple[np.ndarray, List[float]]:
    """
    Performs Q-learning on a FrozenLake environment.

    Args:
        env: Gymnasium FrozenLakeEnv instance.
        Q (np.ndarray): Q-table of shape (n_states, n_actions).
        episodes (int): Number of training episodes.
        max_steps (int): Max steps per episode.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Initial epsilon for epsilon-greedy.
        min_epsilon (float): Floor for epsilon value.
        epsilon_decay (float): Multiplicative decay factor per episode
            applied as: epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    Returns:
        (Q, total_rewards):
            Q (np.ndarray): Updated Q-table.
            total_rewards (List[float]): Sum of rewards per episode.
    """
    total_rewards: List[float] = []

    # Cached map layout to detect holes (so we can convert their reward to -1)
    desc = env.unwrapped.desc  # dtype=bytes, shape=(rows, cols)
    n_rows, n_cols = desc.shape

    def is_hole(state_idx: int) -> bool:
        r, c = divmod(state_idx, n_cols)
        return desc[r, c] == b'H'

    for _ in range(episodes):
        state, _ = env.reset()  # Gymnasium: (obs, info)
        ep_reward = 0.0

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)

            # Convert hole reward to -1 (FrozenLake normally returns 0 on holes)
            if terminated and is_hole(next_state):
                reward = -1.0

            # Q-learning update
            best_next = 0.0 if (terminated or truncated) else np.max(Q[next_state])
            td_target = reward + gamma * best_next
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * td_target

            ep_reward += reward
            state = next_state

            if terminated or truncated:
                break

        total_rewards.append(ep_reward)

        # Decay epsilon after each episode (bounded below by min_epsilon)
        epsilon = max(min_epsilon, epsilon * (1.0 - epsilon_decay))

    return Q, total_rewards
