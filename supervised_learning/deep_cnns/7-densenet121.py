#!/usr/bin/env python3
"""
Builds the DenseNet-121 architecture.
"""
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Conv2D, BatchNormalization, Activation,
                                     MaxPooling2D, GlobalAveragePooling2D, Dense)
from tensorflow.keras.initializers import HeNormal

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer

def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture.

    Args:
        growth_rate: Growth rate for the dense blocks
        compression: Compression factor for transition layers

    Returns:
        The Keras model of DenseNet-121
    """
    initializer = HeNormal(seed=0)
    inputs = Input(shape=(224, 224, 3))

    # Initial convolution and pooling
    X = BatchNormalization()(inputs)
    X = Activation('relu')(X)
    X = Conv2D(64, (7, 7), strides=(2, 2), padding='same', kernel_initializer=initializer)(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(X)

    # Dense blocks + transition layers
    X, nb_filters = dense_block(X, 64, growth_rate, 6)     # 6 conv blocks
    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    # Final layers
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = GlobalAveragePooling2D()(X)
    outputs = Dense(1000, activation='softmax', kernel_initializer=initializer)(X)

    return Model(inputs=inputs, outputs=outputs)
