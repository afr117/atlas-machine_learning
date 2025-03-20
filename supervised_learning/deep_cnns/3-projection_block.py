#!/usr/bin/env python3
"""
Builds a projection block for ResNet as described in
Deep Residual Learning for Image Recognition (2015).
"""

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block.

    Parameters:
    - A_prev: The output of the previous layer.
    - filters: Tuple or list containing F11, F3, F12 respectively:
        - F11: Number of filters in the first 1x1 convolution.
        - F3: Number of filters in the 3x3 convolution.
        - F12: Number of filters in the second 1x1 convolution (and shortcut connection).
    - s: Stride of the first convolution in both the main path and shortcut connection.

    Returns:
    - Activated output of the projection block.
    """

    # Extract filter sizes
    F11, F3, F12 = filters

    # Define HeNormal initializer with fixed seed
    initializer = K.initializers.VarianceScaling(scale=2.0, mode='fan_avg',
                                                 distribution='truncated_normal', seed=0)

    # **MAIN PATH**
    # 1x1 Convolution (Reduce Dimension)
    X = K.layers.Conv2D(filters=F11, kernel_size=(1, 1),
                        strides=s, padding='same',
                        kernel_initializer=initializer)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # 3x3 Convolution
    X = K.layers.Conv2D(filters=F3, kernel_size=(3, 3),
                        strides=1, padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # 1x1 Convolution (Restore Dimension)
    X = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                        strides=1, padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # **SHORTCUT PATH (Projection Shortcut)**
    shortcut = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                               strides=s, padding='same',
                               kernel_initializer=initializer)(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    # **Add Skip Connection**
    X = K.layers.Add()([X, shortcut])
    X = K.layers.Activation('relu')(X)

    return X
