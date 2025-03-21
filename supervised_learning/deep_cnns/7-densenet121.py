#!/usr/bin/env python3
"""
Builds the DenseNet-121 architecture.
"""
from tensorflow import keras as K

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
    initializer = K.initializers.he_normal(seed=0)
    inputs = K.Input(shape=(224, 224, 3))

    # Initial convolution and pooling
    X = K.layers.BatchNormalization()(inputs)
    X = K.layers.ReLU()(X)
    X = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(X)

    # Dense Block 1
    X, nb_filters = dense_block(X, 64, growth_rate, 6)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 2
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 3
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 4
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    # Final layers
    X = K.layers.BatchNormalization()(X)
    X = K.layers.ReLU()(X)
    X = K.layers.GlobalAveragePooling2D()(X)
    outputs = K.layers.Dense(1000, activation='softmax',
                             kernel_initializer=initializer)(X)

    return K.Model(inputs=inputs, outputs=outputs)
