#!/usr/bin/env python3
"""
REINFORCE training loop (Monte-Carlo policy gradient).

Implements a basic episodic policy-gradient update:
    W <- W + alpha * sum_t [ G_t * ∇_W log π(a_t | s_t) ]

- Uses your `policy_gradient(state, weight)` helper to sample an action
  and get the gradient of log π w.r.t. the weight matrix at that state.
- Returns a list of per-episode scores (sum of rewards).
"""

import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Trains a softmax policy with REINFORCE on a Gymnasium environment.

    Args:
        env: initial environment (e.g., gym.make('CartPole-v1')).
        nb_episodes (int): number of training episodes.
        alpha (float): learning rate.
        gamma (float): discount factor.

    Returns:
        list[float]: scores per episode (sum of rewards).
    """
    # Infer dimensions
    obs_space = env.observation_space
    act_space = env.action_space
    state_dim = int(np.prod(getattr(obs_space, "shape", (1,))))
    n_actions = getattr(act_space, "n", None)
    if n_actions is None:
        raise ValueError("This trainer expects a discrete action space with 'n'.")

    # Initialize policy parameters (weights): (state_dim, n_actions)
    # Small random values help initial exploration and numerical stability.
    W = np.random.rand(state_dim, n_actions)

    scores = []

    for ep in range(nb_episodes):
        obs, _ = env.reset()
        obs = np.asarray(obs, dtype=float)

        grads = []     # ∇ log π(a_t | s_t) w.r.t. W, shape (d, k)
        rewards = []   # r_t

        # --- Generate one episode ---
        while True:
            action, grad = policy_gradient(obs, W)
            obs_next, r, terminated, truncated, _ = env.step(action)

            grads.append(grad)
            rewards.append(float(r))

            obs = np.asarray(obs_next, dtype=float)
            if terminated or truncated:
                break

        # Episode score
        score = float(np.sum(rewards))
        scores.append(score)
        print(f"Episode: {ep} Score: {score}")

        # --- Compute discounted returns G_t ---
        G = 0.0
        returns = np.zeros(len(rewards), dtype=float)
        for t in range(len(rewards) - 1, -1, -1):
            G = rewards[t] + gamma * G
            returns[t] = G

        # --- Policy parameter update: accumulate per-time-step grads ---
        update = np.zeros_like(W)
        for g, Gt in zip(grads, returns):
            update += Gt * g  # ascend the gradient

        W += alpha * update

    return scores
