#!/usr/bin/env python3
"""
Builds a transition layer as described in
Densely Connected Convolutional Networks.
"""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer using DenseNet-C compression.

    Parameters:
    - X: Output from the previous layer.
    - nb_filters: Number of filters in X.
    - compression: Compression factor (float between 0 and 1).

    Returns:
    - The output of the transition layer.
    - The number of filters in the output.
    """
    initializer = K.initializers.HeNormal(seed=0)

    # Batch Normalization + ReLU
    BN = K.layers.BatchNormalization(axis=3)(X)
    ACT = K.layers.Activation('relu')(BN)

    # 1x1 Convolution with compression
    compressed_filters = int(nb_filters * compression)
    conv = K.layers.Conv2D(compressed_filters, (1, 1), padding='same',
                           kernel_initializer=initializer)(ACT)

    # 2x2 Average Pooling
    output = K.layers.AveragePooling2D(pool_size=(2, 2), strides=2,
                                       padding='valid')(conv)

    return output, compressed_filters
