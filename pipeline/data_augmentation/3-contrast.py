#!/usr/bin/env python3
"""
Module for image data augmentation: Random Contrast
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image.

    Args:
        image: a 3D tf.Tensor containing the image to adjust.
        lower: float representing lower bound of the contrast factor.
        upper: float representing upper bound of the contrast factor.

    Returns:
        The contrast-adjusted image as a tf.Tensor.
    """
    return tf.image.random_contrast(image, lower, upper)
