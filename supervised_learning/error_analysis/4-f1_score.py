#!/usr/bin/env python3
"""Module to calculate the F1 score of a confusion matrix."""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): A square matrix of shape (classes, classes),
                                   where rows represent the correct labels and
                                   columns represent the predicted labels.

    Returns:
        numpy.ndarray: A 1D array of shape (classes,) containing the F1 score
                       of each class.
    """
    sens = sensitivity(confusion)
    prec = precision(confusion)

    return 2 * (prec * sens) / (prec + sens)
