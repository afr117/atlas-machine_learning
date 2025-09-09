Deep Q-Learning: Atari Breakout (Keras-RL2 + Gymnasium)

This folder contains two scripts to train and play a DQN agent on Atari’s Breakout using keras-rl2, TensorFlow/Keras, and Gymnasium.

train.py — trains a DQN (DQNAgent + SequentialMemory + EpsGreedyQPolicy) and saves the policy weights to policy.h5.

play.py — loads policy.h5, runs with GreedyQPolicy, and plays episodes with either on-screen (human) or headless (rgb_array) rendering.

The code uses Gymnasium wrappers (Atari preprocessing + Step API compatibility) so that keras-rl2 works with the new Gymnasium API.

Quickstart
# from repo root
cd reinforcement_learning/deep_q_learning

1) Install dependencies (CPU-only example)

These versions are known to work together (and match what the scripts expect).

python3 -m pip install -U pip

# Core stack
python3 -m pip install "tensorflow-cpu<2.15" "keras-rl2==1.0.5" "gymnasium==0.29.1"

# Atari support + ROMs + wrappers
python3 -m pip install "gymnasium[atari,other]==0.29.1" "autorom[accept-rom-license]" "shimmy"

# Accept and install the Atari ROMs (runs a small CLI)
AutoROM --accept-license

# Image preproc (used by AtariPreprocessing)
python3 -m pip install "opencv-python-headless==4.9.0.80"

# Numpy known-good with TF 2.12–2.14
python3 -m pip install "numpy>=1.23,<2"


GPU is optional. If CUDA/cuDNN aren’t available you’ll see info/warnings; training still works on CPU.

Train
# Headless/server-friendly (no window)
python3 train.py --steps 50000 --render none

# Small smoke test
python3 train.py --steps 500 --render none


What it does

Builds a CNN policy network (84×84 grayscale frames, stacked window_length=4).

Wraps the environment with AtariPreprocessing (frame skip, grayscale, resize) and Gymnasium’s StepAPICompatibility.

Uses DQNAgent + SequentialMemory + EpsGreedyQPolicy.

Disables TF eager execution internally so keras-rl2’s graph-mode training works with TF-Keras.

Saves final weights to policy.h5 in this folder.

You’ll see logs like:

Training for 50000 steps ...
... episode: 45, ... episode reward: 5.000 ...
Saved weights to policy.h5

Play
# Headless / CI / servers: use rgb_array (no window)
python3 play.py --weights policy.h5 --episodes 2 --render rgb_array

# On a desktop with a display:
python3 play.py --weights policy.h5 --episodes 5 --render human


What it does

Loads policy.h5.

Uses GreedyQPolicy for evaluation.

Runs the requested number of episodes and prints reward/steps.

Example output:

Testing for 2 episodes ...
Episode 1: reward: 0.000, steps: 122
Episode 2: reward: 7.000, steps: 400

CLI Reference
train.py
--steps N                  Number of training steps (int).
--render {none,human,rgb_array}
                           Rendering mode during training.
                           none      : fastest, no frames produced
                           human     : opens a window (needs display)
                           rgb_array : headless-safe frame array

play.py
--weights PATH             Path to saved weights (default: policy.h5)
--episodes N               Number of evaluation episodes (default: 5)
--max-steps N              Optional cap on steps per episode
--render {human,rgb_array} Render mode (use rgb_array on servers)

Notes & gotchas

Gymnasium wrappers:
The scripts create the base env with frameskip=1 and then apply AtariPreprocessing(..., frame_skip=4, ...). This avoids “double” frame skipping errors and yields (84, 84) grayscale observations that stack to (84, 84, 4) for the model.

Headless environments:
If you see Failed to initialize SDL when using --render human, switch to --render rgb_array. On servers there’s no display.

TensorFlow/Keras versions:
keras-rl2 trains in TF1-style graph mode. The script disables eager execution so it runs cleanly on TF 2.x. Stick with TF < 2.15 and Keras < 3 to avoid optimizer API and __version__ issues.

ROM license:
AutoROM --accept-license installs required Atari ROMs for ALE.

Project structure
atlas-machine_learning/
└─ reinforcement_learning/
   └─ deep_q_learning/
      ├─ train.py   # trains DQN, saves policy.h5
      ├─ play.py    # loads policy.h5 and plays episodes
      └─ policy.h5  # produced by training

Troubleshooting

ImportError: cannot import name '__version__' from 'tensorflow.keras'
You likely installed the old keras-rl or a Keras 3.x build. Ensure keras-rl2==1.0.5 and keep Keras < 3 (via TF < 2.15).

AttributeError: 'Adam' object has no attribute 'get_updates'
This happens with newer Keras optimizers in eager mode. The script disables eager execution; if you hit this, verify TF < 2.15 and Keras < 3.

ValueError: Disable frame-skipping in the original env...
Make sure the base env is created with frameskip=1; the wrapper handles the skipping.

Error when checking input: expected frames (84,84,4) but got (4,84,84)
That’s a channel order mismatch. The provided scripts use channels_last ((H, W, C)).

Repro commands used during verification
# Train a quick run
python3 train.py --steps 500 --render none
# -> Saved weights to policy.h5

# Evaluate headlessly
python3 play.py --weights policy.h5 --episodes 2 --render rgb_array
# -> Prints per-episode reward and steps


Happy training!
