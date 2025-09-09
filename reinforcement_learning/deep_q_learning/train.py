#!/usr/bin/env python3
"""
Train a DQN agent on Atari Breakout using keras-rl2 and gymnasium.

- Uses keras-rl2 DQNAgent, SequentialMemory, and EpsGreedyQPolicy.
- Uses Gymnasium wrappers for Atari preprocessing and Step API compatibility.
- Saves the learned policy network as `policy.h5`.

Run:
    python3 train.py --steps 500000
"""

import argparse
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, StepAPICompatibility
import tensorflow as tf
from tensorflow import keras

# ---------- keras-rl2 / TF 2.15 compatibility shim ----------
# keras-rl2 tries: `from tensorflow.keras import __version__ as KERAS_VERSION`
# Ensure the symbol exists on the *module* object 'tensorflow.keras'.
try:
    import sys
    import keras as standalone_keras  # standalone keras package
    try:
        import tensorflow.keras as tfk  # ensure module object is loaded
    except Exception:
        tfk = tf.keras
    if not hasattr(tfk, "__version__"):
        setattr(tfk, "__version__", getattr(standalone_keras, "__version__", "2.15.0"))
    # make sure sys.modules points to the same module carrying __version__
    sys.modules["tensorflow.keras"] = tfk
except Exception:
    pass
# ------------------------------------------------------------

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy


def make_env(render_mode=None):
    """Create Breakout with preprocessing and API compatibility."""
    env = gym.make("ALE/Breakout-v5", render_mode=render_mode)
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


def build_model(window_length, obs_shape, nb_actions):
    """DQN (Nature) conv net: input (wl, 84, 84) → (84,84,wl) → Conv → Dense."""
    wl = window_length
    h, w = obs_shape
    inputs = keras.Input(shape=(wl, h, w))
    x = keras.layers.Permute((2, 3, 1))(inputs)  # (H, W, wl)
    x = keras.layers.Lambda(lambda z: tf.cast(z, tf.float32) / 255.0)(x)
    x = keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation="relu")(x)
    x = keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation="relu")(x)
    x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation="relu")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    outputs = keras.layers.Dense(nb_actions, activation="linear")(x)
    return keras.Model(inputs=inputs, outputs=outputs)


def make_agent(model, nb_actions, window_length):
    """Create and compile a DQNAgent with replay memory and eps-greedy policy."""
    memory = SequentialMemory(limit=1_000_000, window_length=window_length)
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.1,
        value_test=0.05,
        nb_steps=1_000_000,
    )
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=50_000,
        target_model_update=10_000,  # hard updates
        policy=policy,
        gamma=0.99,
        train_interval=4,
        delta_clip=1.0,
        enable_double_dqn=True,
    )
    dqn.compile(optimizer=keras.optimizers.Adam(learning_rate=2.5e-4),
                metrics=["mae"])
    return dqn


def main():
    """CLI entrypoint for training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500_000,
                        help="Training steps.")
    parser.add_argument("--save", type=str, default="policy.h5",
                        help="Model path to save.")
    args = parser.parse_args()

    env = make_env(render_mode=None)
    nb_actions = env.action_space.n
    obs_shape = env.observation_space.shape  # (84, 84)
    window_length = 4

    model = build_model(window_length, obs_shape, nb_actions)
    agent = make_agent(model, nb_actions, window_length)

    agent.fit(env, nb_steps=args.steps, visualize=False, verbose=2)

    # Save the policy network (complete Keras model)
    model.save(args.save)


if __name__ == "__main__":
    main()
