#!/usr/bin/env python3
"""
Module for Temporal Difference TD(0) algorithm.
"""
import numpy as np


def td_zero(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the TD(0) algorithm.

    Args:
        env: gymnasium environment instance.
        V: numpy.ndarray of shape (s,) containing the value estimate.
        policy: function that takes in a state and returns the next action.
        episodes: total number of episodes to train over.
        max_steps: maximum number of steps per episode.
        alpha: learning rate.
        gamma: discount rate.

    Returns:
        V: updated value estimate.
    """
    for _ in range(episodes):
        state, _ = env.reset()
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # TD Update Rule (Bootstrapping)
            # V(s) = V(s) + alpha * [R + gamma * V(s') - V(s)]
            V[state] = V[state] + alpha * (reward + gamma * V[next_state] - V[state])

            if terminated or truncated:
                break
            state = next_state

    return V
