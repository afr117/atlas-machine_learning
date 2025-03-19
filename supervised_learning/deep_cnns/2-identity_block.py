#!/usr/bin/env python3
"""
Builds the Inception Network as described in 'Going Deeper with Convolutions' (2014).
"""

from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds the Inception Network.

    Returns:
    - The Keras model.
    """
    he_normal = K.initializers.HeNormal()

    # Input layer
    X_input = K.Input(shape=(224, 224, 3))

    # Initial Convolution and Pooling layers
    X = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same',
                        activation='relu', kernel_initializer=he_normal)(X_input)
    X = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(X)

    X = K.layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same',
                        activation='relu', kernel_initializer=he_normal)(X)
    X = K.layers.Conv2D(filters=192, kernel_size=(3, 3), padding='same',
                        activation='relu', kernel_initializer=he_normal)(X)
    X = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(X)

    # Inception Blocks
    X = inception_block(X, [64, 96, 128, 16, 32, 32])   # Inception (3a)
    X = inception_block(X, [128, 128, 192, 32, 96, 64]) # Inception (3b)
    X = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(X)

    X = inception_block(X, [192, 96, 208, 16, 48, 64])  # Inception (4a)
    X = inception_block(X, [160, 112, 224, 24, 64, 64]) # Inception (4b)
    X = inception_block(X, [128, 128, 256, 24, 64, 64]) # Inception (4c)
    X = inception_block(X, [112, 144, 288, 32, 64, 64]) # Inception (4d)
    X = inception_block(X, [256, 160, 320, 32, 128, 128]) # Inception (4e)
    X = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(X)

    X = inception_block(X, [256, 160, 320, 32, 128, 128]) # Inception (5a)
    X = inception_block(X, [384, 192, 384, 48, 128, 128]) # Inception (5b)

    # Global Average Pooling
    X = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(X)

    # Dropout Layer
    X = K.layers.Dropout(0.4)(X)

    # Fully Connected Softmax Layer
    X = K.layers.Dense(units=1000, activation='softmax', kernel_initializer=he_normal)(X)

    # Create Model
    model = K.models.Model(inputs=X_input, outputs=X)

    return model
