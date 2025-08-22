# Atlas Machine Learning — Coursework & Labs

![Composite plots](docs/screenshot-atlas-ml.png)

A curated collection of ML coursework and labs: linear algebra utilities, probability & statistics, supervised/unsupervised learning, and deep learning experiments. Each module aims for clear, PEP8 compliant code with docstrings and runnable examples.

## Why this exists (the story)
I built this repo to track my progression from math foundations to applied models. I focused on writing **readable, vectorized** NumPy solutions and small, testable scripts so I could iterate quickly and validate understanding—then captured plots and outputs to show learning milestones.

## Implemented (highlights)
- Linear algebra utilities (slicing, matrix ops, determinants, eigen-stuff)
- Probability & statistics (Poisson/Normal/Binomial classes with PMF/CDF)
- Unsupervised learning (K-means, optimum-k by variance)
- Supervised learning (basic neural nets; object detection exercises)
- Matplotlib plotting suites (line/scatter/stacked bars, multi-subplot figures)
- PEP8 checked with `pycodestyle`; concise docstrings

## To implement (roadmap)
- Unify helpers (I/O, plotting, metrics) in a `utils/` package
- Add `pre-commit` hooks and lightweight CI (lint + tests)
- Dockerfile for turnkey execution
- Expand unit style checks for regression safety
- More experiment notebooks with hyperparameter sweeps

## Hardest challenges
- Enforcing pycodestyle across many small tasks without slowing iteration
- Making vectorized NumPy solutions both **readable** and **fast**
- Organizing modules to avoid duplicate helpers while keeping each task self-contained

## Getting started
```bash
git clone https://github.com/afr117/atlas-machine_learning.git
cd atlas-machine_learning
python -m venv .venv

# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

# If a global requirements file exists:
pip install -r requirements.txt

# Example run (linear algebra slicing task)
python math/linear_algebra/0-slice_me_up.py
Repo structure (excerpt)
lua
Copy
Edit
math/
  linear_algebra/
  probability/
supervised_learning/
  classification/
  object_detection/
unsupervised_learning/
  clustering/
Selected modules
Slice Me Up (math/linear_algebra/0-slice_me_up.py)
Practice Python slicing to extract array subsets without loops or conditionals. The script is exactly 8 lines and follows pycodestyle.

Screenshots
docs/screenshot-atlas-ml.png — a composite of plots/figures from the repo (replace with your own image).

About the developer
I’m Alfred Figueroa—I like turning theory into shippable code.

LinkedIn: https://www.linkedin.com/in/alfred-figueroa-rosado-10b010208

X (Twitter): https://twitter.com/your_handle

Portfolio Project repo: https://github.com/afr117/bussiness

nginx
Copy
Edit

If you want to keep a README **inside** `math/linear_algebra/` specifically for *Slice Me Up*, your draft is fine—just fix the last line and formatting:

```markdown
# Slice Me Up — Python Slicing Task

## Project overview
This task practices Python slicing operations on arrays to extract subsets of data **without** loops or conditionals.

## Task details
- Extract the first two numbers of the array  
- Extract the last five numbers of the array  
- Extract the 2nd through 6th numbers of the array  

**Constraints:** no loops/conditionals; script is exactly 8 lines; pycodestyle compliant.

## Files
- `0-slice_me_up.py` — slicing operations
- `README.md` — this overview

## Requirements
- Python 3.9
- NumPy 1.25.2
- Ubuntu 20.04 LTS
- Files should be executable
- pycodestyle 2.11.1

## How to run
```bash
cd math/linear_algebra
chmod +x 0-slice_me_up.py
./0-slice_me_up.py
