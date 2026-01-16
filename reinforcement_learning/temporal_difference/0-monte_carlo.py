#!/usr/bin/env python3
"""
Module for Monte Carlo algorithm.
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm to estimate the value function.

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
    n = V.shape[0]
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []
        
        # 1. Generate an episode
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            if terminated or truncated:
                break
            state = next_state
            
        # 2. Process the episode
        episode = np.array(episode, dtype=object)
        states = episode[:, 0]
        rewards = episode[:, 1]
        
        G = 0
        discounts = np.array([gamma**i for i in range(len(rewards) + 1)])
        
        # Track visited states in this episode for first-visit MC
        visited_states = set()
        
        # Traverse backward to calculate returns efficiently
        for t in range(len(episode) - 1, -1, -1):
            s_t = episode[t, 0]
            r_t = episode[t, 1]
            G = gamma * G + r_t
            
            # Check if this is the first visit to the state in this episode
            if s_t not in states[:t]:
                V[s_t] = V[s_t] + alpha * (G - V[s_t])
                
    return V
