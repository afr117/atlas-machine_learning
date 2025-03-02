#!/usr/bin/env python3
"""Module to calculate specificity for each class in
a confusion matrix."""

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): A square matrix of shape (classes, classes),
                                   where rows represent the correct labels and
                                   columns represent the predicted labels.

    Returns:
        numpy.ndarray: A 1D array of shape (classes,)
        containing the specificity
                       of each class.
    """
    true_positives = np.diag(confusion)
  # Extract the diagonal (TP)
    false_positives = np.sum(confusion, axis=0) - true_positives
  # FP = Sum of column - TP
    false_negatives = np.sum(confusion, axis=1) - true_positives
  # FN = Sum of row - TP
    true_negatives = np.sum(confusion) - (true_positives + false_positives
                                          + false_negatives) # TN

    return true_negatives / (true_negatives + false_positives)
