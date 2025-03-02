#!/usr/bin/env python3
"""
Calculates the cost of a neural network with L2 regularization.
"""

import numpy as np

def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost (float): Cost of the network without L2 regularization.
        lambtha (float): Regularization parameter.
        weights (dict): Dictionary containing the weights and biases of the network.
        L (int): Number of layers in the neural network.
        m (int): Number of data points used.

    Returns:
        float: The cost of the network accounting for L2 regularization.
    """
    l2_sum = 0
    for i in range(1, L + 1):
        l2_sum += np.linalg.norm(weights[f'W{i}'])**2

    l2_cost = cost + (lambtha / (2 * m)) * l2_sum
    return l2_cost
