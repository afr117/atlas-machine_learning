#!/usr/bin/env python3
"""Bidirectional RNN forward propagation."""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN.

    Args:
        bi_cell: BidirectionalCell instance
        X: input data of shape (t, m, i)
        h_0: initial hidden state for forward direction, shape (m, h)
        h_t: initial hidden state for backward direction, shape (m, h)

    Returns:
        H: concatenated hidden states, shape (t, m, 2h)
        Y: outputs for all timesteps, shape (t, m, o)
    """
    t, m, i = X.shape
    h = h_0.shape[1]

    # Containers for forward/backward hidden states
    H_f = np.zeros((t, m, h))
    H_b = np.zeros((t, m, h))

    # Forward pass (left -> right)
    h_prev = h_0
    for k in range(t):
        h_prev = bi_cell.forward(h_prev, X[k])
        H_f[k] = h_prev

    # Backward pass (right -> left)
    h_next = h_t
    for k in range(t - 1, -1, -1):
        h_next = bi_cell.backward(h_next, X[k])
        H_b[k] = h_next

    # Concatenate forward and backward hidden states
    H = np.concatenate((H_f, H_b), axis=2)  # (t, m, 2h)

    # Outputs via the cell's output layer + softmax
    Y = bi_cell.output(H)  # (t, m, o)

    return H, Y
