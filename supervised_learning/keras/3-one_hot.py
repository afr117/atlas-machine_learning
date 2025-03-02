#!/usr/bin/env python3
"""
Converts a label vector into a one-hot matrix.
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    Args:
        labels (numpy.ndarray): The label vector.
        classes (int, optional): The number of classes.
        If None, inferred automatically.

    Returns:
        numpy.ndarray: The one-hot matrix.
    """
    return K.utils.to_categorical(labels,
                                  num_classes=classes)
