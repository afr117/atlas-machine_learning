#!/usr/bin/env python3
"""
This module contains the function `summation_i_squared`
which calculates the sum of squares of integers from 1 to n.
"""

def summation_i_squared(n):
    """
    Calculates the sum of squares of integers from 1 to n.

    Args:
        n (int): The upper bound integer for the summation.

    Returns:
        int: The sum of squares from 1 to n, or None if n is not valid.

    Example:
        >>> summation_i_squared(5)
        55
    """
    if not isinstance(n, int) or n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
