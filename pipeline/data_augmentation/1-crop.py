#!/usr/bin/env python3
"""
Module for image data augmentation: Random Crop
"""
import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of an image.

    Args:
        image: a 3D tf.Tensor containing the image to crop.
        size: tuple containing size of the crop (height, width, channels).

    Returns:
        The randomly cropped image as a tf.Tensor.
    """
    return tf.image.random_crop(image, size=size)
