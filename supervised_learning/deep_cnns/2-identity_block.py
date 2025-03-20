#!/usr/bin/env python3
"""
Builds an identity block for ResNet as described in Deep Residual Learning for Image Recognition (2015).
"""

from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block.

    Parameters:
    - A_prev: The output from the previous layer.
    - filters: Tuple or list containing F11, F3, F12 respectively:
        - F11: Number of filters in the first 1x1 convolution.
        - F3: Number of filters in the 3x3 convolution.
        - F12: Number of filters in the second 1x1 convolution.

    Returns:
    - Activated output of the identity block.
    """
    F11, F3, F12 = filters

    # Define weight initializer
    initializer = K.initializers.HeNormal(seed=0)

    # First 1x1 Convolution
    X = K.layers.Conv2D(F11, (1, 1), padding="same", kernel_initializer=initializer)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation("relu")(X)

    # 3x3 Convolution
    X = K.layers.Conv2D(F3, (3, 3), padding="same", kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation("relu")(X)

    # Second 1x1 Convolution
    X = K.layers.Conv2D(F12, (1, 1), padding="same", kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Add skip connection (A_prev must have the same shape as X)
    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation("relu")(X)

    return X
