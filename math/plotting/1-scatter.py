#!/usr/bin/env python3
"""
Module 1-scatter

This module contains a function to plot a scatter plot of men's height vs weight.
The x-axis represents height (in inches), and the y-axis represents weight (lbs).
"""

import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """Plots a scatter plot for men's height vs weight."""
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))

    # Plot the scatter plot with magenta points
    plt.scatter(x, y, color='magenta')

    # Add labels and title
    plt.xlabel("Height (in)")
    plt.ylabel("Weight (lbs)")
    plt.title("Men's Height vs Weight")

    # Show the plot
    plt.show()
