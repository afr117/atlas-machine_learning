#!/usr/bin/env python3
"""
Module 3-two

This module contains a function to plot two line graphs representing the
exponential decay of C-14 and Ra-226. The x-axis represents time (years), and
the y-axis represents the fraction remaining.
"""

import numpy as np
import matplotlib.pyplot as plt


def two():
    """Plots two exponential decay graphs for C-14 and Ra-226."""
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730  # Half-life of C-14
    t2 = 1600  # Half-life of Ra-226

    y1 = np.exp((r / t1) * x)  # Decay of C-14
    y2 = np.exp((r / t2) * x)  # Decay of Ra-226

    plt.figure(figsize=(6.4, 4.8))

    # Plot y1 with a dashed red line
    plt.plot(x, y1, 'r--', label='C-14')

    # Plot y2 with a solid green line
    plt.plot(x, y2, 'g-', label='Ra-226')

    # Add labels and title
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.title("Exponential Decay of Radioactive Elements")

    # Set the x-axis and y-axis range
    plt.xlim(0, 20000)
    plt.ylim(0, 1)

    # Add a legend in the upper right corner
    plt.legend(loc='upper right')

    # Show the plot
    plt.show()
