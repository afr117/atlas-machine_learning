#!/usr/bin/env python3
"""
Play one episode on FrozenLake using a trained Q-table.

Returns:
    total_reward (float)
    rendered_outputs (List[str]): board render after reset and after each step
"""
import numpy as np


def _render_to_text(env):
    """
    Converts env.render() output to a plain string for console printing.
    With render_mode='ansi', Gymnasium returns a string (or StringIO-like).
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
        rendered_outputs: List of board states (final state included).
    """
    outputs = []
    total_reward = 0.0

    # Ensure the initial board is in the output
    outputs.append(_render_to_text(env))

    # Get starting state (driver may have already called env.reset())
    if hasattr(env.unwrapped, "s"):
        state = int(env.unwrapped.s)
    else:
        state, _ = env.reset()
        outputs[-1] = _render_to_text(env)

    for _ in range(max_steps):
        # Always exploit the Q-table
        action = int(np.argmax(Q[state]))

        # Step the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        # Append ONLY the board render (the checker prints the action line)
        outputs.append(_render_to_text(env))

        state = int(next_state)
        if terminated or truncated:
            break

    return total_reward, outputs
