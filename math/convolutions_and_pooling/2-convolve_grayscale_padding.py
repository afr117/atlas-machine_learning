#!/usr/bin/env python3
"""
Module to perform a convolution on grayscale images
with custom padding.
"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Parameters:
    images (numpy.ndarray): A numpy array of shape (m, h, w)
    containing multiple grayscale images.
        - m: Number of images.
        - h: Height of images in pixels.
        - w: Width of images in pixels.
    kernel (numpy.ndarray): A numpy array of shape (kh, kw)
    containing the kernel for convolution.
        - kh: Height of the kernel.
        - kw: Width of the kernel.
    padding (tuple): A tuple (ph, pw) representing the
    padding values.
        - ph: Padding for the height of the image.
        - pw: Padding for the width of the image.

    Returns:
    numpy.ndarray: A numpy array containing the
    convolved images.
    """
    # Retrieve dimensions
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Apply zero-padding to images
    padded_images = np.pad(images, ((0, 0), (ph, ph),
                                    (pw, pw)), mode='constant')

    # Compute new dimensions after padding
    new_h = h + 2 * ph - kh + 1
    new_w = w + 2 * pw - kw + 1

    # Initialize output array
    output = np.zeros((m, new_h, new_w))

    # Perform convolution using exactly two for-loops
    for i in range(new_h):
        for j in range(new_w):
            output[:, i, j] = np.sum(padded_images[:, i:i+kh,
                                     j:j+kw] * kernel, axis=(1, 2))

    return output
