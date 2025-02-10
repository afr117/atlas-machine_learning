#!/usr/bin/env python3
"""Defines a function to create the training operation for
a neural network."""

import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network using
    gradient descent.

    Args:
        loss (tf.Tensor): The loss of the network's prediction.
        alpha (float): The learning rate.

    Returns:
        tf.Operation: An operation that trains the network using
        gradient descent.
    """
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
