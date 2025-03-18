#!/usr/bin/env python3
"""
Performs back propagation over a convolutional layer of a neural network.
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer.

    Parameters:
    dZ (numpy.ndarray): Partial derivatives w.r.t the unactivated output of shape
    (m, h_new, w_new, c_new).
    A_prev (numpy.ndarray): Output of the previous layer of shape
    (m, h_prev, w_prev, c_prev).
    W (numpy.ndarray): Kernels of shape (kh, kw, c_prev, c_new).
    b (numpy.ndarray): Bias of shape (1, 1, 1, c_new).
    padding (str): Either 'same' or 'valid', indicating padding type.
    stride (tuple): Tuple of (sh, sw) for stride values.

    Returns:
    dA_prev, dW, db: Gradients w.r.t previous layer, kernels, and biases.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    m, h_new, w_new, c_new = dZ.shape

    if padding == "same":
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1
    else:
        ph, pw = 0, 0

    A_prev_padded = np.pad(A_prev, ((0, 0), (ph, ph),
                                    (pw, pw), (0, 0)), mode='constant')
    dA_prev_padded = np.zeros_like(A_prev_padded)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                slice_A = A_prev_padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
                dA_prev_padded[:, i * sh:i * sh + kh,
                j * sw:j * sw + kw, :] += dZ[:, i, j, k][:, None, None, None] * W[:, :, :, k]
                dW[:, :, :, k] += np.sum(slice_A * dZ[:, i, j, k][:, None, None, None], axis=0)
    
    dA_prev = dA_prev_padded[:, ph:-ph, pw:-pw, :]
    if padding == "same" else dA_prev_padded

    return dA_prev, dW, db
