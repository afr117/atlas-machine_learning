#!/usr/bin/env python3
"""Bidirectional RNN cell (forward step only)."""

import numpy as np


class BidirectionalCell:
    """
    Represents a bidirectional RNN cell.

    Args:
        i (int): dimensionality of the data
        h (int): dimensionality of the hidden state
        o (int): dimensionality of the outputs
    Public attrs:
        Whf, bhf: forward hidden weights/bias, shapes ((i + h), h), (1, h)
        Whb, bhb: backward hidden weights/bias, shapes ((i + h), h), (1, h)
        Wy,  by : output weights/bias, shapes ((2h), o), (1, o)
    """
    def __init__(self, i, h, o):
        """Initialize weights (std normal) and biases (zeros)."""
        # Forward direction params
        self.Whf = np.random.randn(i + h, h)
        self.bhf = np.zeros((1, h))

        # Backward direction params
        self.Whb = np.random.randn(i + h, h)
        self.bhb = np.zeros((1, h))

        # Output params (uses concatenated [h_f, h_b] -> size 2h)
        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Compute next hidden state in the FORWARD direction for one time step.

        Args:
            h_prev (np.ndarray): previous hidden state, shape (m, h)
            x_t (np.ndarray): input at time t, shape (m, i)

        Returns:
            h_next (np.ndarray): next hidden state, shape (m, h)
        """
        concat = np.concatenate((h_prev, x_t), axis=1)  # (m, h + i)
        h_next = np.tanh(np.matmul(concat, self.Whf) + self.bhf)
        return h_next
