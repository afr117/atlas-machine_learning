#!/usr/bin/env python3
"""
Performs forward propagation over a pooling layer of a neural network.
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer.

    Parameters:
    A_prev (numpy.ndarray): Input data of shape (m, h_prev, w_prev, c_prev).
    kernel_shape (tuple): Tuple of (kh, kw) containing kernel size.
    stride (tuple): Tuple of (sh, sw) containing stride values.
    mode (str): Either 'max' or 'avg' for pooling type.

    Returns:
    numpy.ndarray: The output of the pooling layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = (h_prev - kh) // sh + 1
    output_w = (w_prev - kw) // sw + 1

    output = np.zeros((m, output_h, output_w, c_prev))

    for i in range(output_h):
        for j in range(output_w):
            if mode == 'max':
                output[:, i, j, :] = np.max(
                    A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :],
                    axis=(1, 2)
                )
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(
                    A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :],
                    axis=(1, 2)
                )

    return output
