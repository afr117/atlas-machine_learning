#!/usr/bin/env python3
"""
Builds the ResNet-50 architecture as described in Deep Residual Learning for Image Recognition (2015).
"""

from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture.
    
    Returns:
    - Keras model for ResNet-50.
    """

    initializer = K.initializers.HeNormal(seed=0)
    input_layer = K.Input(shape=(224, 224, 3))

    # Initial Conv layer
    X = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same',
                        kernel_initializer=initializer)(input_layer)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.ReLU()(X)  # ✅ FIX: Use ReLU() instead of Activation('relu')
    X = K.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(X)

    # Stage 1: 1 projection block + 2 identity blocks
    X = projection_block(X, [64, 64, 256], s=1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])

    # Stage 2: 1 projection block + 3 identity blocks
    X = projection_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])

    # Stage 3: 1 projection block + 5 identity blocks
    X = projection_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])

    # Stage 4: 1 projection block + 2 identity blocks
    X = projection_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])

    # Average Pooling and Fully Connected layer
    X = K.layers.AveragePooling2D(pool_size=(7, 7), strides=1)(X)
    X = K.layers.Flatten()(X)
    X = K.layers.Dense(units=1000, activation='softmax',
                       kernel_initializer=initializer)(X)

    # Create the model
    model = K.models.Model(inputs=input_layer, outputs=X)

    return model
