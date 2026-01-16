#!/usr/bin/env python3
"""
Module for Monte Carlo algorithm implementation.
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
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []
        
        # 1. Generate an episode following the policy
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            if terminated or truncated:
                break
            state = next_state
            
        # Convert episode to a list for easier indexing
        episode = np.array(episode, dtype=object)
        
        # 2. Calculate Returns (G) and Update V
        G = 0
        # Set to track first visits within this specific episode
        visited_states = set()
        
        # Process the episode in reverse to compute G efficiently
        # G_t = R_{t+1} + gamma * G_{t+1}
        states = episode[:, 0]
        rewards = episode[:, 1]
        
        for t in range(len(episode) - 1, -1, -1):
            s_t = states[t]
            r_t = rewards[t]
            G = gamma * G + r_t
            
            # First-visit check: only update if s_t was not visited earlier
            # in the episode (which, in a reverse loop, means later indices)
            if s_t not in states[:t]:
                V[s_t] = V[s_t] + alpha * (G - V[s_t])
                
    return V
