# Policy Gradients — REINFORCE (Task 0)

Minimal start for a Policy Gradient project.  
This task implements a **softmax policy function**:

policy(matrix, weight) -> softmax(matrix @ weight)

yaml
Copy code

where `matrix` is a batch of state feature vectors and `weight` maps features to action logits.

---

## Requirements
- Ubuntu 20.04 LTS
- Python 3.9
- numpy == 1.25.2
- gymnasium == 0.29.1 (project-wide; not needed for Task 0)
- pycodestyle == 2.11.1

All files:
- start with `#!/usr/bin/env python3`
- end with a newline
- have docstrings and follow pycodestyle

---

## Install (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy==1.25.2 pycodestyle==2.11.1
# (optional, for later tasks)
pip install "gymnasium[toy_text]==0.29.1"
Project Layout
bash
Copy code
atlas-machine_learning/
└─ reinforcement_learning/
   └─ policy_gradients/
      ├─ policy_gradient.py   # Task 0: softmax policy(matrix, weight)
      └─ 0-main.py            # Example runner
Usage
Run the sample:

bash
Copy code
chmod +x reinforcement_learning/policy_gradients/*.py
./reinforcement_learning/policy_gradients/0-main.py
Expected output:

lua
Copy code
[[0.50351642 0.49648358]]
Function Spec
policy(matrix, weight)

Args

matrix: np.ndarray (n, d) — batch of state features

weight: np.ndarray (d, k) — weights to action logits

Returns

np.ndarray (n, k) — softmax probabilities over actions

Details

Uses a numerically stable softmax: subtract max-logit per row before exp.

Style & Docs Checks
bash
Copy code
pycodestyle reinforcement_learning/policy_gradients/
python3 -c 'import policy_gradient as m; print(m.__doc__); print(m.policy.__doc__)'S
