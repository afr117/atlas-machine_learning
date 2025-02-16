#!/usr/bin/env python3
"""
Sets up the RMSProp optimization algorithm in TensorFlow.
"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Sets up the RMSProp optimization algorithm in
    TensorFlow.

    Parameters:
    alpha (float): The learning rate.
    beta2 (float): The RMSProp weight (Discounting factor).
    epsilon (float): A small number to avoid division by zero.

    Returns:
    tensorflow.keras.optimizers.RMSprop:
    The optimizer configured with RMSProp.
    """
    return tf.keras.optimizers.RMSprop(learning_rate=alpha,
                                       rho=beta2, epsilon=epsilon)
