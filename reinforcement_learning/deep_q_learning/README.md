# Deep Q-Learning on Atari Breakout (keras-rl2 + Gymnasium)

This task trains and runs a **Deep Q-Network (DQN)** agent to play
**Atari Breakout** using **keras-rl2** with **Gymnasium** wrappers for API
compatibility and Atari preprocessing.

**Files**
- `train.py` — trains a DQN agent and **saves the policy network** to `policy.h5`
- `play.py` — loads `policy.h5`, uses a **GreedyQPolicy**, and displays a game

---

## Learning Objectives

- What is **Deep Q-learning** and the **policy network** (Q-network)?
- What is **replay memory** and why it stabilizes training?
- What is the **target network** and why we keep it separate?
- How to use **keras-rl2** (`DQNAgent`, `SequentialMemory`, `rl.policy`)
- How to make **Gymnasium** compatible with keras-rl (**wrappers** for `reset`,
  `step`, and `render`)

---

## Requirements & Versions (pinned)

- Python 3.9 (Ubuntu 20.04 LTS)
- `numpy==1.25.2`
- `tensorflow==2.15.0`, `keras==2.15.0`
- `gymnasium==0.29.1` (Atari support via extras)
- `keras-rl2==1.0.4`
- `Pillow==10.3.0`, `h5py==3.11.0`
- `autorom[accept-rom-license]` (auto-install Atari ROMs)

All project files are executable, end with a newline, and follow **pycodestyle
2.11.1**.

---

## Installation

```bash
# Core deps
pip install --user numpy==1.25.2
pip install --user tensorflow==2.15.0 keras==2.15.0
pip install --user gymnasium==0.29.1
pip install --user keras-rl2==1.0.4
pip install --user Pillow==10.3.0 h5py==3.11.0

# Atari env + ROMs
pip install --user "gymnasium[atari]==0.29.1"
pip install --user "autorom[accept-rom-license]"

