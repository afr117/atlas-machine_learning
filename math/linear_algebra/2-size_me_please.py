#!/usr/bin/env python3
"""
Module 2-size_me_please
This module contains a function that calculates
the shape of a matrix.
"""

def matrix_shape(matrix):
    """
    Calculates the shape of a matrix.
    
    Args:
        matrix: A list of lists representing the matrix.

    Returns:
        A list of integers representing the shape of the matrix.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
