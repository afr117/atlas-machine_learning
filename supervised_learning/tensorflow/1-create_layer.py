#!/usr/bin/env python3
"""
Defines a function to create a layer for a neural network
using TensorFlow v1 with He et al. initialization.
"""
import tensorflow.compat.v1 as tf

def create_layer(prev, n, activation):
    """
    Creates a layer for a neural network using He et al. initialization.

    Args:
        prev: Tensor, output of the previous layer.
        n: int, number of nodes in the layer to create.
        activation: Activation function to use for the layer.

    Returns:
        Tensor output of the layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer,
                            name='layer')
    return layer(prev)
