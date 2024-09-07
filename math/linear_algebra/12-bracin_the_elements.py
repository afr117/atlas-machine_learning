#!/usr/bin/env python3
"""
Module 12-bracin_the_elements
This module contains a function that performs element-wise addition,
subtraction, multiplication, and division of two NumPy arrays.
"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication, and
    division of two NumPy arrays.

    Args:
        mat1: A NumPy ndarray.
        mat2: A NumPy ndarray or scalar.

    Returns:
        A tuple containing:
            - The element-wise sum of mat1 and mat2.
            - The element-wise difference of mat1 and mat2.
            - The element-wise product of mat1 and mat2.
            - The element-wise quotient of mat1 and mat2.
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
