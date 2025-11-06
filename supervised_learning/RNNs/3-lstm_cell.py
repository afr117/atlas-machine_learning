#!/usr/bin/env python3
"""Defines a single LSTM cell using NumPy only."""

import numpy as np


class LSTMCell:
    """
    LSTM unit.

    Args:
        i (int): input size
        h (int): hidden size
        o (int): output size

    Public attrs:
        Wf, Wu, Wc, Wo (i+h, h): gate/candidate weights
        Wy (h, o): output weights
        bf, bu, bc, bo (1, h): gate/candidate biases
        by (1, o): output bias
    """

    def __init__(self, i, h, o):
        """Init weights (std normal, in order) and biases (zeros)."""
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
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

    def forward(self, h_prev, c_prev, x_t):
        """
        One time-step forward.

        Args:
            h_prev (m, h): previous hidden state
            c_prev (m, h): previous cell state
            x_t   (m, i): input at time t

        Returns:
            h_next (m, h): next hidden state
            c_next (m, h): next cell state
            y      (m, o): softmax output
        """
        # concat = [h_prev, x_t]
        concat = np.concatenate((h_prev, x_t), axis=1)

        # gates
        f_t = self._sigmoid(np.matmul(concat, self.Wf) + self.bf)  # forget
        u_t = self._sigmoid(np.matmul(concat, self.Wu) + self.bu)  # input
        c_hat = np.tanh(np.matmul(concat, self.Wc) + self.bc)      # cand
        o_t = self._sigmoid(np.matmul(concat, self.Wo) + self.bo)  # output

        # cell and hidden updates
        c_next = f_t * c_prev + u_t * c_hat
        h_next = o_t * np.tanh(c_next)

        # output
        logits = np.matmul(h_next, self.Wy) + self.by
        y = self._softmax(logits)
        return h_next, c_next, y
