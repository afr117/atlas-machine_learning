#!/usr/bin/env python3
"""
REINFORCE training loop (Monte-Carlo policy gradient).

Implements episodic update:
    W <- W + alpha * sum_t [ G_t * ∇_W log π(a_t | s_t) ]

When show_result is True, render every 1000 episodes (0, 1000, 2000, ...)
and the final episode.
"""

import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Train a softmax policy with REINFORCE on a Gymnasium environment.

    Args:
        env: environment, e.g. gym.make('CartPole-v1', render_mode="human").
        nb_episodes (int): number of training episodes.
        alpha (float): learning rate.
        gamma (float): discount factor.
        show_result (bool): render every 1000 episodes if True.

    Returns:
        list[float]: per-episode scores (sum of rewards).
    """
    obs_space = env.observation_space
    act_space = env.action_space
    state_dim = int(np.prod(getattr(obs_space, "shape", (1,))))
    n_actions = getattr(act_space, "n", None)
    if n_actions is None:
        raise ValueError(
            "Discrete action space with attribute 'n' is required."
        )

    W = np.random.rand(state_dim, n_actions)
    scores = []

    for ep in range(nb_episodes):
        obs, _ = env.reset()
        obs = np.asarray(obs, dtype=float)

        grads = []
        rewards = []

        render_this = False
        if show_result:
            if (ep % 1000 == 0) or (ep == nb_episodes - 1):
                render_this = True

        if render_this and ep == 0:
            print("\nResult after few episodes:\n")
        elif render_this and ep == 5000:
            print("\n\nResult after more episodes:\n")
        elif render_this and (ep == nb_episodes - 1):
            msg = f"\n\nResult after {nb_episodes} episodes:\n"
            print(msg)

        while True:
            action, grad = policy_gradient(obs, W)
            if render_this:
                env.render()

            obs_next, r, terminated, truncated, _ = env.step(action)

            grads.append(grad)
            rewards.append(float(r))

            obs = np.asarray(obs_next, dtype=float)
            if terminated or truncated:
                if render_this:
                    env.render()
                break

        score = float(np.sum(rewards))
        scores.append(score)
        print(f"Episode: {ep} Score: {score}")

        G = 0.0
        returns = np.zeros(len(rewards), dtype=float)
        for t in range(len(rewards) - 1, -1, -1):
            G = rewards[t] + gamma * G
            returns[t] = G

        update = np.zeros_like(W)
        for g, Gt in zip(grads, returns):
            update += Gt * g
        W += alpha * update

    return scores
