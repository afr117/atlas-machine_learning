#!/usr/bin/env python3
"""
Module 13-cats_got_your_tongue
This module contains a function that concatenates two NumPy arrays
along a specific axis.
"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two NumPy arrays along a specific axis.

    Args:
        mat1: A NumPy ndarray.
        mat2: A NumPy ndarray.
        axis: The axis along which to concatenate the arrays.

    Returns:
        A new ndarray resulting from concatenation of mat1 and mat2.
    """
    return np.concatenate((mat1, mat2), axis=axis)
