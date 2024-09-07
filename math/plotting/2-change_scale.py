#!/usr/bin/env python3
"""
Module 2-change_scale

This module contains a function to plot the exponential decay of C-14.
The x-axis represents time in years, and the y-axis represents the fraction
remaining. The y-axis is logarithmically scaled.
"""

import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """Plots the exponential decay of C-14 with a logarithmic y-axis."""
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Plot the line graph
    plt.plot(x, y)

    # Add labels and title
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.title("Exponential Decay of C-14")

    # Set y-axis to logarithmic scale
    plt.yscale('log')

    # Set the x-axis range
    plt.xlim(0, 28650)

    # Show the plot
    plt.show()
