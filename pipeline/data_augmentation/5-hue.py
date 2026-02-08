#!/usr/bin/env python3
"""
Module for image data augmentation: Hue Adjustment
"""
import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image.

    Args:
        image: 3D tf.Tensor containing image to change.
        delta: the amount the hue should change.

    Returns:
        The altered image as a tf.Tensor.
    """
    return tf.image.adjust_hue(image, delta)
