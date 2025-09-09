#!/usr/bin/env python3
"""
Train a DQN agent on Atari Breakout with keras-rl2 + Gymnasium.

- Gymnasium→keras-rl2 compatibility wrapper
- Patches keras-rl2 to avoid importing removed Keras __version__
- Disables eager execution (TF1 graph mode) and uses legacy Adam
- Saves weights to policy.h5
"""

import os
import argparse
import importlib
import re
from typing import Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _patch_keras_rl2_for_keras3() -> None:
    """Patch keras-rl2 so it doesn't import removed `tensorflow.keras.__version__`."""
    try:
        import rl  # noqa
        import rl.callbacks as cb  # noqa
        return
    except Exception:
        try:
            rl_pkg = importlib.import_module("rl")
            base = os.path.dirname(rl_pkg.__file__)
            path = os.path.join(base, "callbacks.py")
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read()
            new = re.sub(
                r"from tensorflow\.keras import __version__ as KERAS_VERSION",
                (
                    "try:\n"
                    "    import keras as _k\n"
                    "    KERAS_VERSION = getattr(_k, '__version__', '3')\n"
                    "except Exception:\n"
                    "    KERAS_VERSION = '3'"
                ),
                txt,
                count=1,
            )
            if new != txt:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new)
                importlib.invalidate_caches()
        except Exception:
            pass
        import rl  # noqa
        import rl.callbacks as cb  # noqa


# keras-rl2 expects TF1 graph mode
tf.compat.v1.disable_eager_execution()
_patch_keras_rl2_for_keras3()
from rl.agents.dqn import DQNAgent  # noqa: E402
from rl.policy import EpsGreedyQPolicy  # noqa: E402
from rl.memory import SequentialMemory  # noqa: E402


class KerasRLCompatWrapper:
    """Adapter so Gymnasium env behaves like old Gym for keras-rl2."""
    def __init__(self, env):
        self.env = env

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, _info = out
            return obs
        return out

    def step(self, action):
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
            return obs, reward, done, info
        return out

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()


def get_legacy_adam(lr: float):
    try:
        return keras.optimizers.legacy.Adam(learning_rate=lr)
    except Exception:
        return keras.optimizers.Adam(learning_rate=lr)


def make_env(render_mode: str = "none"):
    import gymnasium as gym
    from gymnasium.wrappers import AtariPreprocessing

    mode = None if render_mode == "none" else render_mode  # 'human' or 'rgb_array'
    env = gym.make("ALE/Breakout-v5", frameskip=1, render_mode=mode)
    env = AtariPreprocessing(
        env,
        screen_size=84,
        grayscale_obs=True,
        frame_skip=4,
        noop_max=30,
        terminal_on_life_loss=False,
        scale_obs=False,
    )
    return KerasRLCompatWrapper(env)


def build_model(window_length: int, obs_shape: Tuple[int, int], nb_actions: int) -> keras.Model:
    # keras-rl2 stacks states as (window, H, W). Build input to match that.
    h, w = obs_shape
    inp = keras.Input(shape=(window_length, h, w), name="frames_channels_first")
    # Move to channels-last for TF conv performance, but use a serializable layer (Permute)
    x = layers.Permute((2, 3, 1), name="to_channels_last")(inp)  # (H, W, window)
    x = layers.Rescaling(1.0 / 255.0)(x)
    x = layers.Conv2D(32, 8, strides=4, activation="relu")(x)
    x = layers.Conv2D(64, 4, strides=2, activation="relu")(x)
    x = layers.Conv2D(64, 3, strides=1, activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    out = layers.Dense(nb_actions, activation="linear", name="q_values")(x)
    return keras.Model(inp, out, name="dqn_breakout")


def make_agent(model: keras.Model, nb_actions: int, window_length: int) -> DQNAgent:
    memory = SequentialMemory(limit=1_000_000, window_length=window_length)
    policy = EpsGreedyQPolicy()
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        memory=memory,
        nb_steps_warmup=50_000,
        gamma=0.99,
        target_model_update=10_000,
        train_interval=4,
        delta_clip=1.0,
    )
    dqn.compile(optimizer=get_legacy_adam(2.5e-4), metrics=["mae"])
    return dqn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DQN on Breakout")
    p.add_argument("--steps", type=int, default=500_000, help="Number of training steps")
    p.add_argument("--render", choices=["none", "human", "rgb_array"], default="none",
                   help="Rendering mode during training")
    p.add_argument("--weights", type=str, default="policy.h5",
                   help="Path to save agent weights (HDF5)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env = make_env(render_mode=args.render)
    nb_actions = env.action_space.n
    obs_shape = env.observation_space.shape  # (84, 84)
    window_length = 4

    model = build_model(window_length, obs_shape, nb_actions)
    agent = make_agent(model, nb_actions, window_length)

    print(f"Training for {args.steps} steps ...")
    agent.fit(env, nb_steps=args.steps, visualize=(args.render == "human"), verbose=2)

    agent.save_weights(args.weights, overwrite=True)
    print(f"Saved weights to {args.weights}")
    env.close()


if __name__ == "__main__":
    main()

