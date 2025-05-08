#!/usr/bin/env python3
"""
This module defines a function to calculate
the adjugate matrix of a square matrix.
"""


def determinant(matrix):
    """
    Helper function to calculate the
    determinant of a matrix.
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
        det += ((-1) ** col) * matrix[0][col] * determinant(sub_matrix)
    return det


def cofactor(matrix):
    """
    Helper function to calculate the cofactor matrix.
    """
    size = len(matrix)
    if size == 1:
        return [[1]]

    cof_matrix = []
    for i in range(size):
        row = []
        for j in range(size):
            sub_matrix = [
                [matrix[r][c] for c in range(size) if c != j]
                for r in range(size) if r != i
            ]
            sign = (-1) ** (i + j)
            cofactor_val = sign * determinant(sub_matrix)
            row.append(cofactor_val)
        cof_matrix.append(row)

    return cof_matrix


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a square matrix.

    Args:
        matrix (list of lists): The matrix to evaluate.

    Returns:
        list of lists: The adjugate matrix.

    Raises:
        TypeError: If input is not a list of lists.
        ValueError: If input is not a non-empty square matrix.
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    cof = cofactor(matrix)

    # Transpose the cofactor matrix to get the adjugate
    adj = [[cof[j][i] for j in range(size)] for i in range(size)]

    return adj
