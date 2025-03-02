#!/usr/bin/env python3
"""Module to calculate L2 regularization cost for a neural network."""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost (float): The cost of the network without L2 regularization.
        lambtha (float): The regularization parameter.
        weights (dict): A dictionary containing the weights and biases of the network.
                        Only weight matrices (keys starting with 'W') are considered.
        L (int): The number of layers in the neural network.
        m (int): The number of data points used.

    Returns:
        float: The cost of the network including L2 regularization.
    """
    l2_penalty = sum(np.linalg.norm(weights[f'W{i}'])**2 for i in range(1, L + 1))
    l2_cost = cost + (lambtha / (2 * m)) * l2_penalty
    return l2_cost
