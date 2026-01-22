Atari Breakout DQN AgentThis project implements a Deep Q-Network (DQN) agent to play Atari's Breakout using keras-rl2, gymnasium, and tensorflow.Project Structuretrain.py: Script to train the DQN agent. It saves the trained weights to policy.h5.play.py: Script to visualize the agent playing the game using the weights from policy.h5.README.md: This documentation file.RequirementsThe code is designed to run on Ubuntu 20.04 LTS with Python 3.9.DependenciesInstall the required packages using the following commands:pip install --user gymnasium[atari]==0.29.1
pip install --user tensorflow==2.15.0
pip install --user keras==2.15.0
pip install --user keras-rl2==1.0.4
pip install --user numpy==1.25.2
pip install --user Pillow==10.3.0
pip install --user h5py==3.11.0
pip install autorom[accept-rom-license]
Usage1. TrainingTo train the agent, run train.py. This will initialize the environment, build the CNN model, and start the training loop../train.py
Note: The training process is set to 100,000 steps for demonstration purposes. For a high-performance agent, this should be increased significantly (e.g., to 10,000,000 steps).Output: policy.h5 (saved model weights).2. PlayingTo see the trained agent in action, run play.py. This loads the weights from policy.h5 and renders the game window../play.py
Implementation DetailsGymnasium Compatibility: A custom GymnasiumWrapper class is implemented to bridge the API differences between the modern gymnasium (reset returns tuple, step returns 5 values) and the older keras-rl2 (expects reset to return observation, step to return 4 values).Preprocessing: AtariPreprocessing is used to resize frames to 84x84, convert to grayscale, and perform frame skipping.Model: The agent uses the standard DeepMind DQN architecture (3 Convolutional layers followed by Dense layers).Policy:Training: EpsGreedyQPolicy with LinearAnnealedPolicy to decay epsilon from 1.0 to 0.1.Playing: GreedyQPolicy to always select the action with the highest Q-value.
