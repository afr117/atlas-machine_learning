#!/usr/bin/env python3
"""
Performs a convolution on grayscale images with optional padding and stride.
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.

    Parameters:
    - images (numpy.ndarray): Shape (m, h, w), containing grayscale images.
      - m: Number of images
      - h: Image height
      - w: Image width
    - kernel (numpy.ndarray): Shape (kh, kw), containing the convolution kernel.
      - kh: Kernel height
      - kw: Kernel width
    - padding (tuple or str): Can be a tuple (ph, pw), 'same', or 'valid'.
      - If 'same', applies zero-padding to maintain the same output size.
      - If 'valid', no padding is applied.
      - If a tuple (ph, pw), applies explicit padding.
    - stride (tuple): (sh, sw), defining the stride along height and width.
      - sh: Stride height
      - sw: Stride width

    Returns:
    - numpy.ndarray: The convolved images.
    """
    # Get image dimensions
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Determine padding
    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
    else:  # Custom tuple (ph, pw)
        ph, pw = padding

    # Apply zero-padding to images
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    # Compute output dimensions
    new_h = (h + 2 * ph - kh) // sh + 1
    new_w = (w + 2 * pw - kw) // sw + 1

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
