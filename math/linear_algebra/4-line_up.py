#!/usr/bin/env python3
"""
Module 4-line_up
This module contains a function that adds two arrays element-wise.
"""

def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise.

    Args:
        arr1: List of ints/floats.
        arr2: List of ints/floats.

    Returns:
        A new list with element-wise sums of arr1 and arr2, or
        None if the lists have different shapes.
    """
    if len(arr1) != len(arr2):
        return None
    return [a + b for a, b in zip(arr1, arr2)]

