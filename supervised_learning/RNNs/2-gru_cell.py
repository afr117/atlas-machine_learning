#!/usr/bin/env python3
"""
GRUCell module: defines a single GRU (Gated Recurrent Unit) cell
implemented with NumPy only.

Specifications:
- Only import numpy as np
- Weights initialized from a standard normal distribution
  in this strict order: Wz, Wr, Wh, Wy
- Biases initialized to zeros in this order: bz, br, bh, by
- Right-side matrix multiplication usage
- Forward pass returns (h_next, y) with softmax output
"""

import numpy as np


class GRUCell:
    """
    Represents a GRU cell.

    Args:
        i (int): dimensionality of the data (input size)
        h (int): dimensionality of the hidden state
        o (int): dimensionality of the outputs

    Public attributes:
        Wz (np.ndarray): update gate weights, shape (i + h, h)
        Wr (np.ndarray): reset gate weights,  shape (i + h, h)
        Wh (np.ndarray): candidate hidden weights, shape (i + h, h)
        Wy (np.ndarray): output weights, shape (h, o)
        bz (np.ndarray): update gate bias, shape (1, h)
        br (np.ndarray): reset gate bias,  shape (1, h)
        bh (np.ndarray): candidate hidden bias, shape (1, h)
        by (np.ndarray): output bias, shape (1, o)
    """

    def __init__(self, i, h, o):
        """Initialize parameters."""
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
        # clip for numerical stability
        x_clip = np.clip(x, -709, 709)  # exp(-709) ~ 1e-308
        return 1.0 / (1.0 + np.exp(-x_clip))

    @staticmethod
    def _softmax(z):
        """Row-wise numerically stable softmax."""
        z_shift = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shift)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step.

        Args:
            h_prev (np.ndarray): previous hidden state, shape (m, h)
            x_t (np.ndarray): input at time t, shape (m, i)

        Returns:
            h_next (np.ndarray): next hidden state, shape (m, h)
            y (np.ndarray): output at time t (softmax), shape (m, o)
        """
        # Concatenate [h_prev, x_t] along feature dimension
        concat = np.concatenate((h_prev, x_t), axis=1)  # shape (m, h + i)

        # Gates
        z_t = self._sigmoid(np.matmul(concat, self.Wz) + self.bz)  # (m, h)
        r_t = self._sigmoid(np.matmul(concat, self.Wr) + self.br)  # (m, h)

        # Candidate hidden state uses reset gate on h_prev
        concat_candidate = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.matmul(concat_candidate, self.Wh) + self.bh)  # (m, h)

        # GRU update (note/correction in prompt):
        # h_next = (1 - z) * h_prev + z * h_tilde
        h_next = (1.0 - z_t) * h_prev + z_t * h_tilde

        # Output
        y_linear = np.matmul(h_next, self.Wy) + self.by  # (m, o)
        y = self._softmax(y_linear)

        return h_next, y
