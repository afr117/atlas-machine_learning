#!/usr/bin/env python3
"""
Module to perform a convolution on grayscale images with padding and stride options.
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images with optional padding and stride.

    Parameters:
    images (numpy.ndarray): A numpy array of shape (m, h, w) containing multiple grayscale images.
        - m: Number of images.
        - h: Height of images in pixels.
        - w: Width of images in pixels.
    kernel (numpy.ndarray): A numpy array of shape (kh, kw) containing the kernel for convolution.
        - kh: Height of the kernel.
        - kw: Width of the kernel.
    padding (tuple or str): Either a tuple (ph, pw), 'same', or 'valid'.
        - If 'same', performs a same convolution.
        - If 'valid', performs a valid convolution.
        - If a tuple:
            - ph: Padding for the height of the image.
            - pw: Padding for the width of the image.
    stride (tuple): A tuple (sh, sw) representing the stride values.
        - sh: Stride for the height of the image.
        - sw: Stride for the width of the image.

    Returns:
    numpy.ndarray: A numpy array containing the convolved images.
    """
    # Retrieve dimensions
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Determine padding
    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
    else:  # Custom padding (ph, pw)
        ph, pw = padding

    # Apply zero-padding to images
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    # Compute new dimensions after applying padding and stride
    new_h = ((h + 2 * ph - kh) // sh) + 1
    new_w = ((w + 2 * pw - kw) // sw) + 1

    # Initialize output array
    output = np.zeros((m, new_h, new_w))

    # Perform convolution using exactly two for-loops
    for i in range(new_h):
        for j in range(new_w):
            output[:, i, j] = np.sum(
                padded_images[:, i * sh: i * sh + kh, j * sw: j * sw + kw] * kernel,
                axis=(1, 2)
            )

    return output
