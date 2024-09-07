#!/usr/bin/env python3
"""
Module 7-gettin_cozy
This module contains a function that concatenates two 2D matrices.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along a specific axis.

    Args:
        mat1: A 2D list of ints/floats.
        mat2: A 2D list of ints/floats.
        axis: The axis along which to concatenate. 0 for rows, 1 for columns.

    Returns:
        A new 2D list with the matrices concatenated, or None if they
        cannot be concatenated due to shape mismatch.
    """
    if axis == 0:
        # Check if the number of columns match
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    
    elif axis == 1:
        # Check if the number of rows match
        if len(mat1) != len(mat2):
            return None
        return [mat1[i] + mat2[i] for i in range(len(mat1))]

    return None  # In case an invalid axis is provided
