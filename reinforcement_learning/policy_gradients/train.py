#!/usr/bin/env python3
"""
REINFORCE training loop (Monte-Carlo policy gradient).

Implements a basic episodic policy-gradient update:
    W <- W + alpha * sum_t [ G_t * ∇_W log π(a_t | s_t) ]

When show_result=True, the environment is rendered every 1000 episodes
(episode indices 0, 1000, 2000, ... and the final episode).
"""

import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Trains a softmax policy with REINFORCE on a Gymnasium environment.

    Args:
        env: initial environment (e.g., gym.make('CartPole-v1',
        render_mode="human")).
        nb_episodes (int): number of training episodes.
        alpha (float): learning rate.
        gamma (float): discount factor.
        show_result (bool): if True,
        render the environment every 1000 episodes.

    Returns:
        list[float]: scores per episode (sum of rewards).
    """
    # Infer dimensions
    obs_space = env.observation_space
    act_space = env.action_space
    state_dim = int(np.prod(getattr(obs_space, "shape", (1,))))
    n_actions = getattr(act_space, "n", None)
    if n_actions is None:
        raise ValueError("This trainer expects a
        discrete action space with 'n'.")

    # Initialize policy parameters (weights): (state_dim, n_actions)
    W = np.random.rand(state_dim, n_actions)

    scores = []

    for ep in range(nb_episodes):
        obs, _ = env.reset()
        obs = np.asarray(obs, dtype=float)

        grads = []
        rewards = []

        # decide whether to render this episode
        render_this_ep = False
        if show_result:
            if (ep % 1000 == 0) or (ep == nb_episodes - 1):
                render_this_ep = True

        # Optional: friendly banners similar to the example script
        if render_this_ep and ep == 0:
            print("\nResult after few episodes:\n")
        elif render_this_ep and (ep == 5000):
            print("\n\nResult after more episodes:\n")
        elif render_this_ep and (ep == nb_episodes - 1):
            print(f"\n\nResult after {nb_episodes} episodes:\n")

        # --- Generate one episode ---
        while True:
            action, grad = policy_gradient(obs, W)
            if render_this_ep:
                env.render()

            obs_next, r, terminated, truncated, _ = env.step(action)

            grads.append(grad)
            rewards.append(float(r))

            obs = np.asarray(obs_next, dtype=float)
            if terminated or truncated:
                # render the final frame as well
                if render_this_ep:
                    env.render()
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

        # --- Policy parameter update ---
        update = np.zeros_like(W)
        for g, Gt in zip(grads, returns):
            update += Gt * g
        W += alpha * update

    return scores
