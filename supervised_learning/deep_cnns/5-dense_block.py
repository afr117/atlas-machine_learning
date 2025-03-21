#!/usr/bin/env python3
"""
Builds a dense block as described in Densely Connected Convolutional Networks.
"""

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block using DenseNet-B bottleneck architecture.

    Parameters:
    - X: Output from the previous layer.
    - nb_filters: Number of filters in X.
    - growth_rate: Growth rate for the dense block.
    - layers: Number of layers in the dense block.

    Returns:
    - The concatenated output of each layer within the dense block.
    - The number of filters within the concatenated outputs.
    """
    initializer = K.initializers.HeNormal(seed=0)

    for i in range(layers):
        # Batch Norm + ReLU
        BN1 = K.layers.BatchNormalization(axis=3)(X)
        ACT1 = K.layers.Activation('relu')(BN1)

        # 1x1 Convolution (Bottleneck)
        conv1 = K.layers.Conv2D(4 * growth_rate, (1, 1), padding='same',
                                kernel_initializer=initializer)(ACT1)

        # Batch Norm + ReLU
        BN2 = K.layers.BatchNormalization(axis=3)(conv1)
        ACT2 = K.layers.Activation('relu')(BN2)

        # 3x3 Convolution
        conv2 = K.layers.Conv2D(growth_rate, (3, 3), padding='same',
                                kernel_initializer=initializer)(ACT2)

        # Concatenate with input
        X = K.layers.Concatenate(axis=3)([X, conv2])

        # Update number of filters
        nb_filters += growth_rate

    return X, nb_filters
