#!/usr/bin/env python3
"""
Module 3-flip_me_over
This module contains a function that returns
the transpose of a 2D matrix.
"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix.

    Args:
        matrix: A 2D list representing the matrix.

    Returns:
        A new 2D list representing the transposed matrix.
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
