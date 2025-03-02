#!/usr/bin/env python3
"""
Calculates the cost of a neural network with L2 regularization.
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the total cost of a neural network with L2 regularization.

    Args:
        cost (tf.Tensor): Tensor containing the cost of the network without L2 regularization.
        model (tf.keras.Model): Keras model including layers with L2 regularization.

    Returns:
        tf.Tensor: Tensor containing the total cost for each layer of the network, accounting for L2 regularization.
    """
    l2_losses = sum(model.losses)
    total_cost = cost + l2_losses
    return total_cost
