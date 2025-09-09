#!/usr/bin/env python3
"""
Play Atari Breakout using a trained DQN policy network (policy.h5).

- Loads the saved policy network from `policy.h5`.
- Uses GreedyQPolicy for action selection (pure exploitation).
- Uses Gymnasium wrappers for Atari preprocessing and Step API compatibility.
- Displays the game window (render_mode='human').

Run:
    python3 play.py --weights policy.h5 --episodes 3
"""

import argparse
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, StepAPICompatibility
import tensorflow as tf
from tensorflow import keras

# ---------- keras-rl2 / TF 2.15 compatibility shim ----------
try:
    import sys
    import keras as standalone_keras
    try:
        import tensorflow.keras as tfk
    except Exception:
        tfk = tf.keras
    if not hasattr(tfk, "__version__"):
        setattr(tfk, "__version__", getattr(standalone_keras, "__version__", "2.15.0"))
    sys.modules["tensorflow.keras"] = tfk
except Exception:
    pass
# ------------------------------------------------------------

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy


def make_env(render_mode="human"):
    """Create Breakout for human rendering and API compatibility."""
    env = gym.make("ALE/Breakout-v5", frameskip=1, render_mode=render_mode)
    env = AtariPreprocessing(
        env,
        screen_size=84,
        grayscale_obs=True,
        frame_skip=4,
        noop_max=30,
        terminal_on_life_loss=False,
        scale_obs=False,
    )
    env = StepAPICompatibility(env, output_truncation_bool=False)
    return env


def main():
    """CLI entrypoint for playing with a greedy policy."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="policy.h5",
                        help="Policy .h5 file.")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to play.")
    parser.add_argument("--max-steps", type=int, default=10000,
                        help="Max steps per episode.")
    args = parser.parse_args()

    env = make_env(render_mode="human")
    nb_actions = env.action_space.n
    obs_shape = env.observation_space.shape  # (84, 84)
    window_length = 4

    # Load the saved policy network
    model = keras.models.load_model(args.weights)

    # Build a DQNAgent shell using GreedyQPolicy for action selection.
    memory = SequentialMemory(limit=10_000, window_length=window_length)
    policy = GreedyQPolicy()
    agent = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=0,
        target_model_update=10_000,
        policy=policy,
        gamma=0.99,
        train_interval=1,
        delta_clip=1.0,
        enable_double_dqn=True,
    )
    agent.compile(optimizer=keras.optimizers.Adam(learning_rate=2.5e-4),
                  metrics=["mae"])

    for _ in range(args.episodes):
        agent.reset_states()

        obs = env.reset()
        if isinstance(obs, tuple):  # StepAPICompatibility may return (obs, info)
            obs = obs[0]

        done = False
        steps = 0
        total_reward = 0.0

        while not done and steps < args.max_steps:
            env.render()
            action = agent.forward(obs)          # greedy action
            obs, reward, done, info = env.step(action)
            agent.backward(0.0, terminal=done)   # keep internal state updated
            total_reward += float(reward)
            steps += 1

        print("Episode reward:", total_reward)

    env.close()


if __name__ == "__main__":
    main()
