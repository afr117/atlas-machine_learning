#!/usr/bin/env python3
"""
Module 14-saddle_up
This module contains a function that performs matrix multiplication.
"""

import numpy as np


def np_matmul(mat1, mat2):
    """
    Performs matrix multiplication of two NumPy arrays.

    Args:
        mat1: A NumPy ndarray.
        mat2: A NumPy ndarray.

    Returns:
        A new ndarray resulting from matrix multiplication of mat1 and mat2.
    """
    return np.matmul(mat1, mat2)
