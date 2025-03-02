#!/usr/bin/env python3
"""Module to calculate precision for each class in
a confusion matrix."""

import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): A square matrix of shape (classes, classes),
                                   where rows represent the correct labels and
                                   columns represent the predicted labels.

    Returns:
        numpy.ndarray: A 1D array of shape (classes,) containing the precision
                       of each class.
    """
    true_positives = np.diag(confusion)  # Extract the diagonal (TP)
    false_positives = np.sum(confusion, axis=0) - true_positives
    # Sum of column - TP
    return true_positives / (true_positives + false_positives)
