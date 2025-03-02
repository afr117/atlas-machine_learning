#!/usr/bin/env python3
"""Module to calculate sensitivity for each class in a confusion matrix."""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity (recall) for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): A square matrix of shape (classes, classes),
                                   where rows represent the correct labels and
                                   columns represent the predicted labels.

    Returns:
        numpy.ndarray: A 1D array of shape (classes,)
        containing the sensitivity
                       of each class.
    """
    true_positives = np.diag(confusion)  # Extract the diagonal (TP)
    false_negatives = np.sum(confusion, axis=1)
    - true_positives # Sum of row - TP
    return true_positives / (true_positives + false_negatives)
