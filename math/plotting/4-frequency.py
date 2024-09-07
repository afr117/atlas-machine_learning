#!/usr/bin/env python3
"""
Module 4-frequency

This module contains a function to plot a histogram of student grades.
The x-axis represents grades, and the y-axis represents the number of students.
"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Plots a histogram of student grades for Project A."""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Plot the histogram with bins every 10 units and black outlines for the bars
    plt.hist(student_grades, bins=np.arange(0, 101, 10), edgecolor='black')

    # Set the x-axis and y-axis labels
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")

    # Set the title
    plt.title("Project A")

    # Display the plot
    plt.show()
