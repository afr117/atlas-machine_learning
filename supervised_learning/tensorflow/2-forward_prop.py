#!/usr/bin/env python3
"""
Defines a function for creating the forward propagation graph for a neural network.
"""

import tensorflow.compat.v1 as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.

    Args:
        x: tf.placeholder, the input data.
        layer_sizes: list, number of nodes in each layer.
        activations: list, activation functions for each layer.

    Returns:
        Tensor: The prediction of the network.
    """
    layer = x
    for i in range(len(layer_sizes)):
        activation = activations[i] if activations[i] is not None else None
        layer = create_layer(layer, layer_sizes[i], activation)
    return layer
