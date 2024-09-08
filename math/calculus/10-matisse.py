#!/usr/bin/env python3
"""
This module contains the function `poly_derivative`
which calculates the derivative of a polynomial.
"""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial.

    Args:
        poly (list): A list of coefficients representing the polynomial.
                     The index of the list represents the power of x that
                     the coefficient belongs to.

    Returns:
        list: A list of coefficients representing the derivative of the
              polynomial. Returns [0] if the derivative is zero, or None
              if the input is invalid.

    Example:
        >>> poly_derivative([5, 3, 0, 1])
        [3, 0, 3]
    """
    if not isinstance(poly, list) or len(poly) == 0 or not all(isinstance(coef, (int, float)) for coef in poly):
        return None
    if len(poly) == 1:
        return [0]

    derivative = [i * poly[i] for i in range(1, len(poly))]
    return derivative if derivative else [0]
