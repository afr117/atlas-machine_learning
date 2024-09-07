#!/usr/bin/env python3
"""
Module 0-line

This module contains a function to plot y = x^3 as a solid red line
with the x-axis ranging from 0 to 10.
"""

import numpy as np
import matplotlib.pyplot as plt

def line():
    """Plots y = x^3 as a solid red line from x = 0 to 10."""
    y = np.arange(0, 11) ** 3
    x = np.arange(0, 11)

    plt.plot(x, y, color='red', linestyle='-')
    plt.xlim(0, 10)  # Set x-axis range
    plt.show()
