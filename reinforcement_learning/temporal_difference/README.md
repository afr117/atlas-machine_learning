# Temporal Difference — Monte Carlo (RL)

Minimal implementation of **Monte Carlo state-value prediction** for a Gymnasium environment (e.g., `FrozenLake8x8-v1`).  
Task: `0-monte_carlo.py` exposes `monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99)`.

## Requirements
- Ubuntu 20.04 LTS
- Python 3.9
- numpy == 1.25.2
- gymnasium == 0.29.1 (with toy_text envs)
- pycodestyle == 2.11.1

## Install
```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy==1.25.2 "gymnasium[toy_text]==0.29.1" pycodestyle==2.11.1

