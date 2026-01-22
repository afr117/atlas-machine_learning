#!/usr/bin/env python3
"""
Module to play Atari Breakout using a trained DQN agent.
"""
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
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
    model.add(Permute((2, 3, 1), input_shape=input_shape))
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
    Main function to load policy and play the game.
    """
    # Create the environment with render mode for visualization
    env_name = 'BreakoutNoFrameskip-v4'
    # render_mode='human' allows us to see the game window
    env = gym.make(env_name, render_mode='human')
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True,
                             scale_obs=True)
    env = GymnasiumWrapper(env)

    nb_actions = env.action_space.n
    window_length = 4
    input_shape = (window_length,) + env.observation_space.shape

    # Build model (structure must match training)
    model = build_model(input_shape, nb_actions)

    # Configure Memory
    memory = SequentialMemory(limit=10000, window_length=window_length)

    # Configure Agent with Greedy Policy for testing
    policy = GreedyQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy,
                   memory=memory, gamma=0.99)
    # Compile dummy agent to load weights (optimizer not needed for testing)
    dqn.compile(optimizer='adam')

    # Load weights
    try:
        dqn.load_weights('policy.h5')
        print("Loaded weights from policy.h5")
    except OSError:
        print("Error: policy.h5 not found. Run train.py first.")
        return

    # Play the game
    print(f"Playing {env_name} for 5 episodes...")
    dqn.test(env, nb_episodes=5, visualize=True)


if __name__ == '__main__':
    main()
