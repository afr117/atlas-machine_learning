#!/usr/bin/env python3
"""
This module defines a function to calculate the minor matrix of a square matrix.
"""

def determinant(matrix):
    """
    Helper function to calculate determinant of a matrix (copied from 0-determinant.py)
    """
    if matrix == [[]]:
        return 1

    size = len(matrix)
    if size == 1:
        return matrix[0][0]
    if size == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    det = 0
    for col in range(size):
        sub_matrix = [
            [matrix[i][j] for j in range(size) if j != col]
            for i in range(1, size)
        ]
        sign = (-1) ** col
        det += sign * matrix[0][col] * determinant(sub_matrix)
    return det

def minor(matrix):
    """
    Calculates the minor matrix of a square matrix.

    Args:
        matrix (list of lists): The matrix to evaluate.

    Returns:
        list of lists: The minor matrix.

    Raises:
        TypeError: If input is not a list of lists.
        ValueError: If input is not a non-empty square matrix.
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if size == 1:
        return [[1]]

    minor_matrix = []
    for i in range(size):
        row = []
        for j in range(size):
            sub_matrix = [
                [matrix[r][c] for c in range(size) if c != j]
                for r in range(size) if r != i
            ]
            row.append(determinant(sub_matrix))
        minor_matrix.append(row)

    return minor_matrix
