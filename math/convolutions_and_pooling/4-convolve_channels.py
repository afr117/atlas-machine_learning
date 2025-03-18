#!/usr/bin/env python3
"""
Performs convolution on images with multiple channels.
"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels.

    Parameters:
    images (numpy.ndarray): Array of shape (m, h, w, c)
    containing multiple images.
    kernel (numpy.ndarray): Array of shape (kh, kw, c)
    containing the kernel for the convolution.
    padding (tuple, str): Either a tuple of (ph, pw),
    'same', or 'valid'.
    stride (tuple): Tuple of (sh, sw)
    representing stride values.

    Returns:
    numpy.ndarray: The convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    padded_h = h + 2 * ph
    padded_w = w + 2 * pw
    output_h = (padded_h - kh) // sh + 1
    output_w = (padded_w - kw) // sw + 1

    images_padded = np.pad(images, ((0, 0), (ph, ph),
                                    (pw, pw), (0, 0)), mode='constant')
    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(
                images_padded[:, i * sh:i * sh + kh,
                j * sw:j * sw + kw] * kernel,
                axis=(1, 2, 3)
            )

    return output
