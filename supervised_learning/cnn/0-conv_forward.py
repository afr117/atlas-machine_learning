#!/usr/bin/env python3
"""
Performs forward propagation over a
convolutional layer of a neural network.
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer.

    Parameters:
    A_prev (numpy.ndarray): Input data of shape
    (m, h_prev, w_prev, c_prev).
    W (numpy.ndarray): Weights of shape
    (kh, kw, c_prev, c_new).
    b (numpy.ndarray): Bias of shape
    (1, 1, 1, c_new).
    activation (function): Activation function applied to
    the convolution.
    padding (str): Either 'same' or 'valid',
    indicating the type of padding used.
    stride (tuple): Tuple of (sh, sw) containing stride values.

    Returns:
    numpy.ndarray: The output of the convolutional layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1
    elif padding == "valid":
        ph, pw = 0, 0
    else:
        raise ValueError("Padding must be 'same' or 'valid'")

    padded_h = h_prev + 2 * ph
    padded_w = w_prev + 2 * pw
    output_h = (padded_h - kh) // sh + 1
    output_w = (padded_w - kw) // sw + 1

    A_prev_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')
    output = np.zeros((m, output_h, output_w, c_new))

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j, :] = activation(
                np.sum(A_prev_padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :, np.newaxis] * W,
                       axis=(1, 2, 3)) + b
            )

    return output
