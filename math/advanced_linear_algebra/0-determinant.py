#!/usr/bin/env python3
"""
This module defines the function to calculate the determinant of a matrix.
"""


def determinant(matrix):
    """
    Calculates the determinant of a square matrix.

    Args:
        matrix (list of lists): The matrix to evaluate.

    Returns:
        int or float: The determinant of the matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square.
    """
    # Type and shape validation
    if not isinstance(matrix, list)
    or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    size = len(matrix)
    if not all(len(row) == size for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base cases
    if size == 0:
        return 1
    if size == 1:
        return matrix[0][0]
    if size == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    # Recursive case (Laplace expansion)
    det = 0
    for col in range(size):
        sub_matrix = [
            [matrix[i][j] for j in range(size) if j != col]
            for i in range(1, size)
        ]
        sign = (-1) ** col
        det += sign * matrix[0][col] * determinant(sub_matrix)
    return det
