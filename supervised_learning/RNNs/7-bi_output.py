#!/usr/bin/env python3
"""Bidirectional RNN cell with forward/backward steps and output mapping."""

import numpy as np


class BidirectionalCell:
    """
    Represents a bidirectional RNN cell.

    Args:
        i (int): dimensionality of the data
        h (int): dimensionality of the hidden state
        o (int): dimensionality of the outputs

    Public attrs:
        Whf, bhf: forward hidden weights/bias,  shapes ((i + h), h), (1, h)
        Whb, bhb: backward hidden weights/bias, shapes ((i + h), h), (1, h)
        Wy,  by : output weights/bias,          shapes ((2h), o), (1, o)
    """
    def __init__(self, i, h, o):
        """Initialize weights from N(0,1) and biases to zeros."""
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
        Next hidden state in the FORWARD direction for one time step.

        h_prev: (m, h)
        x_t:    (m, i)
        returns: h_next (m, h)
        """
        concat = np.concatenate((h_prev, x_t), axis=1)  # (m, h + i)
        h_next = np.tanh(concat @ self.Whf + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        Previous hidden state in the BACKWARD direction for one time step.

        h_next: (m, h)  # the 'next' hidden state when scanning right->left
        x_t:    (m, i)
        returns: h_prev (m, h)
        """
        concat = np.concatenate((h_next, x_t), axis=1)  # (m, h + i)
        h_prev = np.tanh(concat @ self.Whb + self.bhb)
        return h_prev

    def output(self, H):
        """
        Compute all outputs for the RNN over time.

        H: (t, m, 2h) concatenated hidden states (forward || backward)
        returns: Y of shape (t, m, o) after softmax
        """
        t, m, _ = H.shape
        # Linear map per time step: (m, 2h) @ (2h, o) + (1, o) -> (m, o)
        Z = H @ self.Wy + self.by  # (t, m, o)

        # Stable softmax along the last axis
        Z_shift = Z - Z.max(axis=2, keepdims=True)
        expZ = np.exp(Z_shift)
        Y = expZ / expZ.sum(axis=2, keepdims=True)
        return Y
