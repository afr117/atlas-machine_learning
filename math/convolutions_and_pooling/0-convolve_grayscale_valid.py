#!/usr/bin/env python3
"""
Module to perform a valid convolution on grayscale images.
"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Parameters:
    images (numpy.ndarray): A numpy array of shape (m, h, w) containing multiple grayscale images.
        - m: Number of images.
        - h: Height of images in pixels.
        - w: Width of images in pixels.
    kernel (numpy.ndarray): A numpy array of shape (kh, kw) containing the kernel for convolution.
        - kh: Height of the kernel.
        - kw: Width of the kernel.

    Returns:
    numpy.ndarray: A numpy array containing the convolved images.
    """
    # Retrieve dimensions
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Compute output dimensions
    new_h = h - kh + 1
    new_w = w - kw + 1

    # Initialize output array
    output = np.zeros((m, new_h, new_w))

    # Perform valid convolution using two for-loops only
    for i in range(new_h):
        for j in range(new_w):
            output[:, i, j] = np.sum(images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))

    return output
