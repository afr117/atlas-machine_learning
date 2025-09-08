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
        env: FrozenLakeEnv instance (created with render_mode='ansi').
        Q: Trained Q-table (numpy.ndarray of shape (n_states, n_actions)).
        max_steps: Max steps per episode.

    Returns:
        total_reward: Sum of rewards obtained in the episode.
        rendered_outputs: List of board states and single action labels
                          between states (final state included).
    """
    action_names = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
    outputs = []
    total_reward = 0.0

    # initial board
    outputs.append(_render_to_text(env))

    # get starting state (env.reset() already called by driver)
    if hasattr(env.unwrapped, "s"):
        state = int(env.unwrapped.s)
    else:
        state, _ = env.reset()
        outputs[-1] = _render_to_text(env)

    for _ in range(max_steps):
        # exploit only
        action = int(np.argmax(Q[state]))

        # step
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        # action label then board
        outputs.append("  ({})".format(action_names.get(action, str(action))))
        outputs.append(_render_to_text(env))

        state = int(next_state)
        if terminated or truncated:
            break

    # ---- remove consecutive duplicate lines (fix double action labels) ----
    rendered_outputs = []
    last = None
    for line in outputs:
        if line != last:
            rendered_outputs.append(line)
        last = line
    # ----------------------------------------------------------------------

    return total_reward, rendered_outputs
