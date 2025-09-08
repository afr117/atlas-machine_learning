#!/usr/bin/env python3
"""
Play one episode on FrozenLake using a trained Q-table.
"""

import numpy as np
from typing import List, Tuple


def _render_to_text(env) -> str:
    """
    Converts env.render() output to a plain string for console printing.
    Gymnasium with render_mode='ansi' returns a string.
    """
    out = env.render()
    # Handle potential edge cases (StringIO or list), though Gymnasium returns str.
    if hasattr(out, "getvalue"):
        return out.getvalue().rstrip()
    if isinstance(out, (list, tuple)):
        return "\n".join(map(str, out)).rstrip()
    return str(out).rstrip()


def play(env, Q: np.ndarray, max_steps: int = 100) -> Tuple[float, List[str]]:
    """
    Runs a greedy (exploit-only) episode using Q on the given env.

    Args:
        env: FrozenLakeEnv instance (already created with render_mode='ansi').
        Q (np.ndarray): Trained Q-table of shape (n_states, n_actions).
        max_steps (int): Max steps per episode.

    Returns:
        (total_reward, rendered_outputs)
            total_reward (float): Sum of rewards obtained in the episode.
            rendered_outputs (List[str]): Sequence of console-rendered states and
                action labels between states. The final state is included.
    """
    # Map FrozenLake actions to readable names
    action_names = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}

    rendered_outputs: List[str] = []
    total_reward = 0.0

    # IMPORTANT: Caller .reset()'s the env in 4-main.py; don't reset here.
    # Capture and store the initial state rendering.
    rendered_outputs.append(_render_to_text(env))

    # Extract current state; Gymnasium returns state in .unwrapped.s or via last step.
    # Safest: track state through step; if starting fresh, we can get it by finding the agent.
    # However, FrozenLake tracks state internally; the API surface does not expose it directly.
    # We will derive state by stepping with a no-op approach? Not possible. Instead,
    # the standard pattern is to reset() before play and capture the returned observation.
    # Since 4-main.py does env.reset() before calling play(), fetch it from last reset via .np_random?
    # Easiest: do a harmless call to env.unwrapped.s if available.
    if hasattr(env.unwrapped, "s"):
        state = int(env.unwrapped.s)
    else:
        # Fallback: reset to get the starting observation (keeps behavior correct)
        state, _ = env.reset()
        rendered_outputs[-1] = _render_to_text(env)  # refresh initial rendering

    for _ in range(max_steps):
        # Always exploit: choose the action with the highest Q-value
        action = int(np.argmax(Q[state]))

        # Take the step
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        # Append the action label and the resulting board
        rendered_outputs.append(f"  ({action_names.get(action, str(action))})")
        rendered_outputs.append(_render_to_text(env))

        state = int(next_state)
        if terminated or truncated:
            break

    return total_reward, rendered_outputs
