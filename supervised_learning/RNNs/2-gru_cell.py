#!/usr/bin/env python3
"""Defines a single GRU (Gated Recurrent Unit) cell using NumPy only."""

import numpy as np


class GRUCell:
    """
    GRU cell.

    Args:
        i (int): input size
        h (int): hidden size
        o (int): output size

    Public attrs:
        Wz, Wr, Wh (i+h, h): gate/candidate weights
        Wy (h, o): output weights
        bz, br, bh (1, h): gate/candidate biases
        by (1, o): output bias
    """

    def __init__(self, i, h, o):
        """Initialize weights (std normal) and biases (zeros)."""
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def _sigmoid(x):
        """Numerically stable sigmoid."""
        x = np.clip(x, -709, 709)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _softmax(z):
        """Row-wise softmax."""
        z = z - np.max(z, axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / np.sum(ez, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        One time-step forward.

        Args:
            h_prev (m, h): previous hidden state
            x_t   (m, i): input at time t

        Returns:
            h_next (m, h): next hidden state
            y      (m, o): softmax output
        """
        # concat = [h_prev, x_t]
        concat = np.concatenate((h_prev, x_t), axis=1)

        # gates
        z_t = self._sigmoid(np.matmul(concat, self.Wz) + self.bz)
        r_t = self._sigmoid(np.matmul(concat, self.Wr) + self.br)

        # candidate uses reset on h_prev
        cand_in = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.matmul(cand_in, self.Wh) + self.bh)

        # GRU update: (1 - z)*h_prev + z*h_tilde
        h_next = (1.0 - z_t) * h_prev + z_t * h_tilde

        # output
        logits = np.matmul(h_next, self.Wy) + self.by
        y = self._softmax(logits)
        return h_next, y
