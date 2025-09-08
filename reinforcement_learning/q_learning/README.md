# Reinforcement Learning – Q-Learning

This project explores **reinforcement learning concepts** using environments from the [Gymnasium](https://gymnasium.farama.org/) library.  
Each task builds toward implementing and understanding Q-Learning and related algorithms.

---

## 📂 Directory
`reinforcement_learning/q_learning/`

---

## 🚀 Task 0: Load the Environment

### Objective
Write a function `load_frozen_lake(desc=None, map_name=None, is_slippery=False)` that loads the **FrozenLake-v1** environment from Gymnasium.

- **Arguments:**
  - `desc`: `None` or a list of lists describing a custom map.
  - `map_name`: `None` or a string specifying a pre made map (`'4x4'`, `'8x8'`, etc.).
  - `is_slippery`: Boolean to determine if the ice is slippery (`False` for deterministic, `True` for stochastic).

- **Returns:**
  - The FrozenLake environment object.

If both `desc` and `map_name` are `None`, a random 8x8 map is generated.

### Example (from `0-main.py`)
```python
env = load_frozen_lake()
print(env.unwrapped.desc)

env = load_frozen_lake(is_slippery=True)
print(env.unwrapped.desc)

desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
print(env.unwrapped.desc)

env = load_frozen_lake(map_name='4x4')
print(env.unwrapped.desc)
