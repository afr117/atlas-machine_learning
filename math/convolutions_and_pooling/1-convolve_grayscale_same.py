#!/usr/bin/env python3
"""
Module to perform a same convolution on grayscale images.
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    Parameters:
    images (numpy.ndarray): A numpy array of shape (m, h, w) containing multiple grayscale images.
        - m: Number of images.
        - h: Height of images in pixels.
        - w: Width of images in pixels.
    kernel (numpy.ndarray): A numpy array of shape (kh, kw) containing the kernel for convolution.
        - kh: Height of the kernel.
        - kw: Width of the kernel.

    Returns:
    numpy.ndarray: A numpy array containing the convolved images with the same spatial dimensions.
    """
    # Retrieve dimensions
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding needed to maintain the same size
    pad_h = kh // 2
    pad_w = kw // 2

    # Pad the images with zeros
    padded_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    # Initialize output array
    output = np.zeros((m, h, w))

    # Perform same convolution using exactly two for-loops
    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(padded_images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))

    return output
