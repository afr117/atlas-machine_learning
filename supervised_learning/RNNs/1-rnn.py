#!/usr/bin/env python3
"""
Module that defines a function for forward propagation of a simple RNN.
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN.

    Args:
        rnn_cell: is an instance of RNNCell that will be used for the
                  forward propagation.
        X (np.ndarray): is the data to be used, given as a numpy.ndarray
                        of shape (t, m, i)
                        t is the maximum number of time steps
                        m is the batch size
                        i is the dimensionality of the data
        h_0 (np.ndarray): is the initial hidden state, given as a
                          numpy.ndarray of shape (m, h)
                          h is the dimensionality of the hidden state

    Returns:
        H (np.ndarray): containing all of the hidden states
        Y (np.ndarray): containing all of the outputs
    """
    # Get dimensions
    t, m, i = X.shape
    m, h = h_0.shape
    o = rnn_cell.Wy.shape[1]  # Output dimensionality

    # Initialize arrays to store hidden states (H) and outputs (Y)

    # H must include initial hidden state h_0, so its shape is (t + 1, m, h)
    # The first slice (H[0]) will hold h_0.
    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    # Y holds the outputs at each time step, so its shape is (t, m, o)
    Y = np.zeros((t, m, o))

    # Start with the initial hidden state
    h_prev = h_0

    # Iterate through each time step
    for step in range(t):
        # Current input data for this time step, shape (m, i)
        x_t = X[step]

        # Perform forward propagation using the RNN cell
        # rnn_cell.forward returns (h_next, y)
        h_next, y_t = rnn_cell.forward(h_prev, x_t)

        # Store the next hidden state and the current output
        H[step + 1] = h_next
        Y[step] = y_t

        # Update the hidden state for the next time step
        h_prev = h_next

    return H, Y

