#!/usr/bin/env python3
"""
Calculates the weighted moving average of a dataset with bias correction.
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a
    dataset using bias correction.

    Parameters:
    data (list): List of data points to
    calculate the moving average of.
    beta (float): Weight used for the moving average.

    Returns:
    list: A list containing the moving averages of data.
    """
    moving_averages = []
    v = 0
    for t, value in enumerate(data, 1):
        v = beta * v + (1 - beta) * value
        corrected_v = v / (1 - beta ** t)
        moving_averages.append(corrected_v)

    return moving_averages
