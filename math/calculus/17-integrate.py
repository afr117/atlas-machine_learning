#!/usr/bin/env python3
"""
This module contains the function `poly_integral`
which calculates the integral of a polynomial.
"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.

    Args:
        poly (list): A list of coefficients representing the polynomial.
                     The index of the list represents the power of x.
        C (int): The constant of integration. Default is 0.

    Returns:
        list: A list of coefficients representing the integral of the
              polynomial, or None if poly or C are invalid.

    Example:
        >>> poly_integral([5, 3, 0, 1])
        [0, 5, 1.5, 0, 0.25]
    """
    if not isinstance(poly, list) or not all(isinstance(coef, (int, float)) for coef in poly):
        return None
    if not isinstance(C, int):
        return None

    # Start with the constant of integration C
    integral = [C]

    # Calculate the integral for each term
    for i in range(len(poly)):
        coef = poly[i] / (i + 1)
        # If the coefficient is a whole number, store it as an integer
        if coef.is_integer():
            coef = int(coef)
        integral.append(coef)

    # Remove trailing zeros to make the list as small as possible
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
