#!/usr/bin/env python3
"""
Creates a layer of a neural network using dropout.
"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.

    Args:
        prev (tf.Tensor): Output tensor from the previous layer.
        n (int): Number of nodes the new layer should contain.
        activation (callable): Activation function for the new layer.
        keep_prob (float): Probability that a node will be kept.
        training (bool): Whether the model is in training mode.

    Returns:
        tf.Tensor: Output tensor of the new layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                        mode='fan_avg')
    regularizer = tf.keras.regularizers.L2(l2=1-keep_prob)
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer
    )(prev)

    if training:
        layer = tf.keras.layers.Dropout(rate=1-keep_prob)
        (layer, training=training)

    return layer
