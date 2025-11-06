#!/usr/bin/env python3

import numpy as np

class GRUCell:
    """
    Represents a Gated Recurrent Unit (GRU) cell.
    """
    def __init__(self, i, h, o):
        """
        Initializes the GRU Cell weights and biases.

        Args:
            i (int): Dimensionality of the data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        # Weights for the update gate (z): Wz (i+h, h)
        # Weights will be used on the right side for matrix multiplication.
        # Wz will multiply [x_t, h_prev] -> (m, i+h) @ (i+h, h) -> (m, h)
        self.Wz = np.random.randn(i + h, h)
        
        # Weights for the reset gate (r): Wr (i+h, h)
        self.Wr = np.random.randn(i + h, h)
        
        # Weights for the intermediate hidden state (h_tilde): Wh (i+h, h)
        self.Wh = np.random.randn(i + h, h)
        
        # Weights for the output (y): Wy (h, o)
        # Wy will multiply h_next -> (m, h) @ (h, o) -> (m, o)
        self.Wy = np.random.randn(h, o)

        # Biases for the gates and states (initialized to zeros)
        # Biases are (1, h) for z, r, h_tilde, and (1, o) for y
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """Helper function for the sigmoid activation."""
        return 1.0 / (1.0 + np.exp(-x))

    def softmax(self, x):
        """Helper function for the softmax activation."""
        # Subtract max for numerical stability
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): Shape (m, h) containing the previous hidden state.
            x_t (numpy.ndarray): Shape (m, i) containing the data input for the cell.

        Returns:
            tuple: (h_next, y)
                h_next (numpy.ndarray): The next hidden state (m, h).
                y (numpy.ndarray): The output of the cell (m, o).
        """
        # Concatenate x_t and h_prev along the feature axis (axis 1)
        # Resulting shape is (m, i + h)
        combined_input = np.concatenate((h_prev, x_t), axis=1)

        # --- 1. Update Gate (z_t) ---
        # z_t = sigmoid(x_t @ Wxz + h_prev @ Whz + bz)
        # Since the weights Wz were initialized as (i+h, h), 
        # the computation is Wz for [h_prev, x_t]
        z_t = self.sigmoid(combined_input @ self.Wz + self.bz)

        # --- 2. Reset Gate (r_t) ---
        # r_t = sigmoid(x_t @ Wxr + h_prev @ Whr + br)
        r_t = self.sigmoid(combined_input @ self.Wr + self.br)

        # --- 3. Intermediate Hidden State (h_tilde) ---
        # h_tilde = tanh(x_t @ Wx_h_tilde + (r_t * h_prev) @ Wh_h_tilde + bh)
        # The GRU formulation requires the input to h_tilde to be [x_t, r_t * h_prev]
        # In this implementation, the weights Wh are (i+h, h).
        # We need to construct the weighted input for Wh: [h_prev * r_t, x_t] 
        # based on the initialization order [h_prev, x_t] for Wz and Wr.
        
        # New combined input for h_tilde: [r_t * h_prev, x_t]
        # Since our initialization of Wh is (i+h, h) and we are multiplying with 
        # [h_prev, x_t] to get (m, h), we need to split Wh into W_h_h and W_x_h
        # and re-order the matrix multiplication:
        # h_tilde = tanh(x_t @ Wx_h_tilde + (r_t * h_prev) @ Wh_h_tilde + bh)

        # For Wz and Wr: [h_prev, x_t] @ W = [h_prev @ W1 + x_t @ W2] (if W is split)
        # If Wh is (i+h, h), where rows 0 to h-1 correspond to h_prev and rows h to i+h-1 
        # correspond to x_t (based on numpy.concatenate((h_prev, x_t)) order):
        # Wh[:h, :] = Wh_h_tilde (h, h)
        # Wh[h:, :] = Wx_h_tilde (i, h)
        
        # Since the weights are for [h_prev, x_t], 
        # we can compute: h_prev_weighted = (r_t * h_prev) @ Wh[:h, :]
        # and x_t_weighted = x_t @ Wh[h:, :]
        
        # Simplified for a generic (i+h, h) Wh:
        # The 'mandatory' requirement states the weights are in the order Wz, Wr, Wh, Wy, 
        # and the weights should be initialized using a random normal distribution in that order.
        # The key is to correctly apply the reset gate to h_prev before computing h_tilde.

        # Let's re-examine the concatenation and weights:
        # combined_input for z and r is (h_prev @ W_h + x_t @ W_x) + b
        
        # For h_tilde, the standard GRU calculation is:
        # h_tilde_activation = (r_t * h_prev) @ Wh_h + x_t @ Wx_h + bh
        
        # We need to compute (r_t * h_prev) @ Wh_h_tilde + x_t @ Wx_h_tilde + bh
        
        # Splitting Wh into two parts: 
        # W_h_h: hidden state part of Wh (h, h)
        # W_x_h: input data part of Wh (i, h)
        W_h_h = self.Wh[:h, :]
        W_x_h = self.Wh[h:, :]

        h_tilde = np.tanh(
            (r_t * h_prev) @ W_h_h + x_t @ W_x_h + self.bh
        )

        # --- 4. Next Hidden State (h_next) ---
        # h_next = z_t * h_prev + (1 - z_t) * h_tilde
        h_next = z_t * h_prev + (1 - z_t) * h_tilde

        # --- 5. Output (y) ---
        # y = softmax(h_next @ Wy + by)
        y = self.softmax(h_next @ self.Wy + self.by)

        return h_next, y


