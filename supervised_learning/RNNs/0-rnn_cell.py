#!/usr/bin/env python3
"""
Module that defines a class for a simple RNN cell.
"""
import numpy as np


class RNNCell:
    """
    Represents a cell of a simple Recurrent Neural Network (RNN).
    """

    def __init__(self, i, h, o):
        """
        Constructor for RNNCell.

        i: is the dimensionality of the data
        h: is the dimensionality of the hidden state
        o: is the dimensionality of the outputs
        """
        # Weights for the hidden state calculation (Wh) and output (Wy)
        # Weights must be initialized using a random normal distribution.
        # Wh is for the concatenated hidden state (h) and input data (i), so its shape is (i + h, h)
        self.Wh = np.random.randn(i + h, h)

        # Wy is for the output, shape (h, o)
        self.Wy = np.random.randn(h, o)

        # Biases for the hidden state (bh) and output (by)
        # Biases must be initialized as zeros.
        # bh is for the hidden state, shape (1, h)
        self.bh = np.zeros((1, h))

        # by is for the output, shape (1, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.
        The activation function for the hidden state is tanh.
        The activation function for the output is softmax.

        Args:
            h_prev (np.ndarray): shape (m, h) containing the previous hidden state.
            x_t (np.ndarray): shape (m, i) containing the data input for the cell.
                              m is the batche size for the data.

        Returns:
            h_next (np.ndarray): the next hidden state.
            y (np.ndarray): the output of the cell.
        """
        # 1. Combine the previous hidden state (h_prev) and current input (x_t)
        # Concatenate along axis 1 (columns) to get shape (m, i + h)
        h_x = np.concatenate((h_prev, x_t), axis=1)

        # 2. Calculate the next hidden state (h_next)
        # h_next = tanh(h_x @ Wh + bh)
        # Matrix multiplication is h_x @ Wh
        h_next_raw = np.matmul(h_x, self.Wh) + self.bh
        h_next = np.tanh(h_next_raw)

        # 3. Calculate the output (y)
        # y_raw = h_next @ Wy + by
        y_raw = np.matmul(h_next, self.Wy) + self.by

        # Apply softmax activation to get the output probability distribution
        # Softmax formula: exp(x_i) / sum(exp(x_j))
        y_exp = np.exp(y_raw)
        y = y_exp / np.sum(y_exp, axis=1, keepdims=True)

        return h_next, y
