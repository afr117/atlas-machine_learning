#!/usr/bin/env python3
"""
Training a DQN agent to play Breakout using keras-rl2 and gymnasium.
"""
import gymnasium as gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, Permute
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy


class GymWrapper(gym.Wrapper):
    """
    Wraps Gymnasium environment to be compatible with keras-rl2.
    """
    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        return res[0]

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        return state, reward, done or truncated, info


def create_model(window, actions):
    """
    Creates the CNN model for Atari Breakout.
    """
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(window, 84, 84)))
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


if __name__ == "__main__":
    env = gym.make("ALE/Breakout-v5")
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    env = GymWrapper(env)

    nb_actions = env.action_space.n
    model = create_model(4, nb_actions)

    memory = SequentialMemory(limit=1000000, window_length=4)
    policy = EpsGreedyQPolicy()

    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
                   nb_steps_warmup=50000, target_model_update=10000,
                   policy=policy)
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

    dqn.fit(env, nb_steps=1750000, visualize=False, verbose=2)
    dqn.save_weights('policy.h5', overwrite=True)
