#!/usr/bin/env python3
"""
Module to train a DQN agent on Atari Breakout using Keras-RL2 and Gymnasium.
"""
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
import numpy as np


class GymnasiumWrapper(gym.Wrapper):
    """
    A wrapper to make Gymnasium environments compatible with Keras-RL2.

    Keras-RL2 expects the old Gym API:
    - reset() returns just the observation.
    - step() returns (observation, reward, done, info).
    """

    def __init__(self, env):
        """
        Initialize the wrapper.

        Args:
            env: The gymnasium environment to wrap.
        """
        super().__init__(env)
        self.env = env

    def reset(self, **kwargs):
        """
        Reset the environment and return the initial observation.

        Returns:
            observation: The initial observation.
        """
        obs, _ = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action: The action to take.

        Returns:
            observation: The new observation.
            reward: The reward obtained.
            done: Whether the episode is finished.
            info: Additional information.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated or truncated, info


def build_model(input_shape, nb_actions):
    """
    Build the Convolutional Neural Network (CNN) model for the DQN agent.

    Args:
        input_shape: The shape of the input observations.
        nb_actions: The number of possible actions.

    Returns:
        model: A compiled Keras model.
    """
    model = Sequential()
    # Keras-RL passes input as (batch, window, height, width) for grayscale
    # Permute to (batch, height, width, window) for TF Conv2D
    model.add(Permute((2, 3, 1), input_shape=input_shape))

    # DeepMind architecture
    model.add(Conv2D(32, (8, 8), strides=(4, 4)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    return model


def main():
    """
    Main function to setup environment, model, and training loop.
    """
    # Create the environment with preprocessing
    # BreakoutNoFrameskip-v4 is standard for DQN
    env_name = 'BreakoutNoFrameskip-v4'
    env = gym.make(env_name)
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True,
                             scale_obs=True, terminal_on_life_loss=True)
    env = GymnasiumWrapper(env)

    nb_actions = env.action_space.n
    # SequentialMemory handles frame stacking (Window Length 4)
    window_length = 4
    input_shape = (window_length,) + env.observation_space.shape

    # Build model
    model = build_model(input_shape, nb_actions)
    print(model.summary())

    # Configure Memory
    memory = SequentialMemory(limit=1000000, window_length=window_length)

    # Configure Policy (Epsilon Greedy with Annealing)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                  value_max=1.0, value_min=0.1,
                                  value_test=0.05, nb_steps=1000000)

    # Configure Agent
    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy,
                   memory=memory, nb_steps_warmup=50000,
                   gamma=0.99, target_model_update=10000,
                   train_interval=4, delta_clip=1.0)

    # Compile Agent
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

    # Train Agent
    # Note: nb_steps set lower for demonstration; increase for full convergence
    print(f"Training on {env_name}...")
    dqn.fit(env, nb_steps=100000, visualize=False, verbose=2)

    # Save the final policy
    dqn.save_weights('policy.h5', overwrite=True)
    print("Policy saved to policy.h5")


if __name__ == '__main__':
    main()
