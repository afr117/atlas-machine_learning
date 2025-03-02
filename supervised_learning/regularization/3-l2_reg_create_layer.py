#!/usr/bin/env python3
"""
Creates a neural network layer with L2 regularization in TensorFlow.
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a TensorFlow layer with L2 regularization.

    Args:
        prev (tf.Tensor): Tensor containing the output of the previous layer.
        n (int): Number of nodes in the layer.
        activation (callable): Activation function to be used on the layer.
        lambtha (float): L2 regularization parameter.

    Returns:
        tf.Tensor: Output of the new layer.
    """
    regularizer = tf.keras.regularizers.L2(lambtha)
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg")
    layer = tf.keras.layers.Dense(n, activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer)
    return layer(prev)
