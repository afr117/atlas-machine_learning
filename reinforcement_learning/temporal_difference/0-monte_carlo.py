#!/usr/bin/env python3
"""
Module for Monte Carlo algorithm
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm.

    Args:
        env: The gymnasium environment instance.
        V: numpy.ndarray of shape (s,) containing the value estimate.
        policy: A function that takes in a state and returns the next action
                to take.
        episodes: Total number of episodes to train over.
        max_steps: The maximum number of steps per episode.
        alpha: The learning rate.
        gamma: The discount rate.

    Returns:
        V: The updated value estimate.
    """
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []

        # Generate an episode
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            if terminated or truncated:
                break
            state = next_state

        # Convert episode to states and rewards for easier indexing
        states = [step[0] for step in episode]
        rewards = [step[1] for step in episode]

        # Calculate returns and update V
        G = 0
        # Iterate backwards through the episode
        for i in range(len(episode) - 1, -1, -1):
            s = states[i]
            r = rewards[i]
            G = gamma * G + r

            # First-visit check: verify if state 's' appeared before index 'i'
            if s not in states[:i]:
                V[s] = V[s] + alpha * (G - V[s])

    return V
