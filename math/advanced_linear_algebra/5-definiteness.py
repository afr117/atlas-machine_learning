#!/usr/bin/env python3
"""
This module defines a function to determine the
definiteness of a square matrix.
"""

import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix using its eigenvalues.

    Args:
        matrix (np.ndarray): The matrix to evaluate.

    Returns:
        str or None: The type of definiteness, or None if undefined.

    Raises:
        TypeError: If matrix is not a numpy.ndarray.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if (matrix.ndim != 2 or 
        matrix.shape[0] != matrix.shape[1] or matrix.size == 0:)
        return None

    if not np.allclose(matrix, matrix.T):
        return None  # Not symmetric => definiteness undefined

    eigvals = np.linalg.eigvalsh(matrix)

    if np.all(eigvals > 0):
        return "Positive definite"
    if np.all(eigvals >= 0):
        return "Positive semi-definite"
    if np.all(eigvals < 0):
        return "Negative definite"
    if np.all(eigvals <= 0):
        return "Negative semi-definite"
    if np.any(eigvals > 0) and np.any(eigvals < 0):
        return "Indefinite"

    return None
