#!/usr/bin/env python3
"""
Module 6-bars

This module contains a function to plot a stacked bar graph showing the number
of different types of fruits various people possess.
The bars are stacked, representing apples, bananas, oranges, and peaches.
"""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Plots a stacked bar graph of the number of fruit per person."""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))

    # Define the labels and colors for the fruits
    labels = ['Farrah', 'Fred', 'Felicia']
    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]

    # Define the figure size
    plt.figure(figsize=(6.4, 4.8))

    # Create the stacked bar plot
    plt.bar(labels, apples, width=0.5, color='red', label='apples')
    plt.bar(labels, bananas, width=0.5, bottom=apples,
            color='yellow', label='bananas')
    plt.bar(labels, oranges, width=0.5, bottom=apples + bananas,
            color='#ff8000', label='oranges')
    plt.bar(labels, peaches, width=0.5, bottom=apples + bananas + oranges,
            color='#ffe5b4', label='peaches')

    # Add y-axis label, title, and legend
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.legend()

    # Set the y-axis limits and ticks
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))

    # Show the plot
    plt.show()
