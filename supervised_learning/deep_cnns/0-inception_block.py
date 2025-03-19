#!/usr/bin/env python3
"""
Builds an inception block as described in Going Deeper with Convolutions (2014).
"""

from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Builds an Inception block.

    Parameters:
    - A_prev: Output from the previous layer.
    - filters: Tuple of filter sizes (F1, F3R, F3, F5R, F5, FPP).

    Returns:
    - The concatenated output of the inception block.
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    he_normal = K.initializers.HeNormal()

    # 1x1 Convolution Branch
    conv1x1 = K.layers.Conv2D(filters=F1, kernel_size=(1, 1), padding='same',
                              activation='relu', kernel_initializer=he_normal)(A_prev)

    # 1x1 Convolution -> 3x3 Convolution Branch
    conv3x3_reduce = K.layers.Conv2D(filters=F3R, kernel_size=(1, 1), padding='same',
                                     activation='relu', kernel_initializer=he_normal)(A_prev)
    conv3x3 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                              activation='relu', kernel_initializer=he_normal)(conv3x3_reduce)

    # 1x1 Convolution -> 5x5 Convolution Branch
    conv5x5_reduce = K.layers.Conv2D(filters=F5R, kernel_size=(1, 1), padding='same',
                                     activation='relu', kernel_initializer=he_normal)(A_prev)
    conv5x5 = K.layers.Conv2D(filters=F5, kernel_size=(5, 5), padding='same',
                              activation='relu', kernel_initializer=he_normal)(conv5x5_reduce)

    # Max Pooling -> 1x1 Convolution Branch
    max_pool = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(A_prev)
    conv_pool = K.layers.Conv2D(filters=FPP, kernel_size=(1, 1), padding='same',
                                activation='relu', kernel_initializer=he_normal)(max_pool)

    # Concatenation of all branches
    output = K.layers.Concatenate(axis=-1)([conv1x1, conv3x3, conv5x5, conv_pool])

    return output
