#!/usr/bin/env python3
"""
Displaying a game played by the trained DQN agent.
"""
import gymnasium as gym
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
# Import helper functions from train
from train import create_model, GymWrapper


if __name__ == "__main__":
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    env = GymWrapper(env)

    nb_actions = env.action_space.n
    model = create_model(4, nb_actions)

    memory = SequentialMemory(limit=1000000, window_length=4)
    policy = GreedyQPolicy()

    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
                   policy=policy)
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

    dqn.load_weights('policy.h5')
    dqn.test(env, nb_episodes=10, visualize=True)
