#!/usr/bin/env python3
"""Forward propagation for a deep RNN (stack of RNNCell layers)."""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN.

    Args:
        rnn_cells (list): list of length l with RNNCell instances.
        X (np.ndarray): input data of shape (t, m, i).
        h_0 (np.ndarray): initial hidden states of shape (l, m, h).

    Returns:
        H (np.ndarray): all hidden states, shape (t + 1, l, m, h)
                        with H[0] == h_0.
        Y (np.ndarray): outputs for each time step from last layer,
                        shape (t, m, o).
    """
    t, m, _ = X.shape
    l, _, h = h_0.shape

    # Infer output dimension from the last cell's Wy
    o = rnn_cells[-1].Wy.shape[1]

    # Allocate containers
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0
    Y = np.zeros((t, m, o))

    # Time loop
    for time in range(t):
        x_t = X[time]
        h_prev_layer = H[time]  # shape (l, m, h)

        # Propagate through layers
        layer_input = x_t
        for layer_idx, cell in enumerate(rnn_cells):
            h_prev = h_prev_layer[layer_idx]
            h_next, y = cell.forward(h_prev, layer_input)
            H[time + 1, layer_idx] = h_next
            layer_input = h_next  # input to next layer is current h

        # Output from the last layer's forward (y from final cell)
        Y[time] = y

    return H, Y
