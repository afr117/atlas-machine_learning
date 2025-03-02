#!/usr/bin/env python3
"""Module to create a confusion matrix."""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Args:
        labels (numpy.ndarray): One-hot numpy array of shape (m, classes)
                                containing the correct labels.
        logits (numpy.ndarray): One-hot numpy array of shape (m, classes)
                                containing the predicted labels.

    Returns:
        numpy.ndarray: Confusion matrix of shape (classes, classes),
                       where rows represent correct labels and
                       columns represent predicted labels.
    """
    return np.matmul(labels.T, logits)
