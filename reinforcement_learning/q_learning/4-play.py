#!/usr/bin/env python3
"""
Play one episode on FrozenLake using a trained Q-table.
"""

import numpy as np


def _render_to_text(env):
    """
    Converts env.render() output to a plain string for console printing.
    Gymnasium with render_mode='ansi' returns a string.
    """
    out = env.render()
    if hasattr(out, "getvalue"):
        return out.getvalue().rstrip()
    if isinstance(out, (list, tuple)):
        return "\n".join(map(str, out)).rstrip()
    return str(out).rstrip()


def play(env, Q, max_steps=100):
    """
    Runs a greedy (exploit-only) episode using Q on the given env.

    Args:
        env: FrozenLakeEnv instance (already created with render_mode='ansi').
        Q: Trained Q-table (numpy.ndarray of shape (n_states, n_actions)).
        max_steps: Max steps per episode.

    Returns:
        total_reward: Sum of rewards obtained in the episode.
        rendered_outputs: List of console-rendered states and
            action labels between states (final state included).
    """
    action_names = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
    rendered_outputs = []
    total_reward = 0.0

    # Capture initial state rendering
    rendered_outputs.append(_render_to_text(env))

    # Get starting state from env internals if available
    if hasattr(env.unwrapped, "s"):
        state = int(env.unwrapped.s)
    else:
        state, _ = env.reset()
        rendered_outputs[-1] = _render_to_text(env)

    for _ in range(max_steps):
        # Exploit: choose the action with the highest Q-value
        action = int(np.argmax(Q[state]))

        # Step through environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        # Append action + board rendering
        rendered_outputs.append("  ({})".format(action_names.get(action, str(action))))
        rendered_outputs.append(_render_to_text(env))

        state = int(next_state)
        if terminated or truncated:
            break

    return total_reward, rendered_outputs
