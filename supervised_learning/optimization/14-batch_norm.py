#!/usr/bin/env python3
"""
Creates a batch normalization layer for
a neural network in TensorFlow.
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for
    a neural network in TensorFlow.

    Parameters:
    prev (tensor): The activated output of the previous layer.
    n (int): The number of nodes in the layer to be created.
    activation (function): The activation function that should
    be used on the output of the layer.

    Returns:
    tensor: The activated output for the layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(units=n,
                                  kernel_initializer=initializer)(prev)

    mean, variance = tf.nn.moments(dense, axes=[0])
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)
    epsilon = 1e-7

    batch_norm = tf.nn.batch_normalization(dense, mean, variance,
                                           beta, gamma, epsilon)

    return activation(batch_norm)
