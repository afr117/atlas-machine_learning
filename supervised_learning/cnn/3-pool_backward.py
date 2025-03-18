#!/usr/bin/env python3
"""
Performs back propagation over a pooling layer of a neural network.
"""
import numpy as np

def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer.

    Parameters:
    dA (numpy.ndarray): Partial derivatives w.r.t the output of the pooling layer (m, h_new, w_new, c).
    A_prev (numpy.ndarray): Output of the previous layer (m, h_prev, w_prev, c).
    kernel_shape (tuple): Size of the kernel for pooling (kh, kw).
    stride (tuple): Tuple of (sh, sw) for stride values.
    mode (str): Either 'max' or 'avg', indicating pooling type.

    Returns:
    dA_prev: Gradients w.r.t previous layer.
    """
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    m, h_new, w_new, c = dA.shape
    
    dA_prev = np.zeros_like(A_prev)
    
    for i in range(h_new):
        for j in range(w_new):
            for k in range(c):
                slice_A = A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, k]
                if mode == 'max':
                    mask = (slice_A == np.max(slice_A, axis=(1, 2), keepdims=True))
                    dA_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, k] += mask * dA[:, i, j, k][:, np.newaxis, np.newaxis]
                elif mode == 'avg':
                    avg_gradient = dA[:, i, j, k][:, np.newaxis, np.newaxis] / (kh * kw)
                    dA_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, k] += avg_gradient
    
    return dA_prev
