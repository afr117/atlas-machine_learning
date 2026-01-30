#!/usr/bin/env python3
"""
Module for SARSA(lambda) algorithm.
"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs SARSA(lambda) reinforcement learning algorithm.

    Args:
        env: The gymnasium environment instance.
        Q: numpy.ndarray of shape (s, a) containing the Q table.
        lambtha: The eligibility trace factor.
        episodes: Total number of episodes to train over.
        max_steps: Maximum number of steps per episode.
        alpha: The learning rate.
        gamma: The discount rate.
        epsilon: Initial threshold for epsilon-greedy.
        min_epsilon: Minimum value for epsilon.
        epsilon_decay: Decay rate for epsilon between episodes.

    Returns:
        Q: The updated Q table.
    """
    # Initial epsilon for the first episode
    init_epsilon = epsilon
    num_states, num_actions = Q.shape

    for ep in range(episodes):
        # Reset state and eligibility traces for each episode
        state, _ = env.reset()
        E = np.zeros((num_states, num_actions))

        # Select initial action using epsilon-greedy
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(Q[state])

        for _ in range(max_steps):
            # Take action, observe reward and next state
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Select next action (SARSA is on-policy)
            if np.random.uniform(0, 1) < epsilon:
                next_action = np.random.randint(num_actions)
            else:
                next_action = np.argmax(Q[next_state])

            # Calculate TD error (delta)
            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]

            # Update eligibility trace for the current state-action pair
            # Using accumulating traces
            E[state, action] += 1

            # Update Q table and Eligibility traces for all states
            Q += alpha * delta * E
            E *= gamma * lambtha

            if terminated or truncated:
                break

            state = next_state
            action = next_action

        # Decay epsilon
        epsilon = min_epsilon + (init_epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * ep)

    return Q
