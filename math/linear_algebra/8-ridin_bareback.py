#!/usr/bin/env python3
"""
Module 8-ridin_bareback
This module contains a function that performs matrix multiplication.
"""


def mat_mul(mat1, mat2):
    """
    Performs matrix multiplication of two 2D matrices.

    Args:
        mat1: A 2D list of ints/floats.
        mat2: A 2D list of ints/floats.

    Returns:
        A new 2D list representing the result of matrix multiplication,
        or None if the matrices cannot be multiplied.
    """
    # Check if the number of columns in mat1 matches the number of rows in mat2
    if len(mat1[0]) != len(mat2):
        return None

    # Perform matrix multiplication
    result = []
    for i in range(len(mat1)):
        row_result = []
        for j in range(len(mat2[0])):
            # Calculate dot product of mat1[i] and the column mat2[:,j]
            dot_product = sum(mat1[i][k] * mat2[k][j] for k in range(len(mat2)))
            row_result.append(dot_product)
        result.append(row_result)

    return result
