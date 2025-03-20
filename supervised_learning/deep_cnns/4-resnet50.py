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
    - Keras model of ResNet-50.
    """

    # He Normal Initialization
    initializer = K.initializers.HeNormal(seed=0)

    # Input Layer
    X_input = K.Input(shape=(224, 224, 3))

    # Initial Convolution & MaxPooling
    X = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="same",
                        kernel_initializer=initializer)(X_input)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)  # FIXED: Changed from ReLU() to Activation('relu')
    X = K.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")(X)

    # Stage 1
    X = projection_block(X, [64, 64, 256], s=1)  
    X = identity_block(X, [64, 64, 256])  
    X = identity_block(X, [64, 64, 256])  

    # Stage 2
    X = projection_block(X, [128, 128, 512], s=2)  
    X = identity_block(X, [128, 128, 512])  
    X = identity_block(X, [128, 128, 512])  
    X = identity_block(X, [128, 128, 512])  

    # Stage 3
    X = projection_block(X, [256, 256, 1024], s=2)  
    X = identity_block(X, [256, 256, 1024])  
    X = identity_block(X, [256, 256, 1024])  
    X = identity_block(X, [256, 256, 1024])  
    X = identity_block(X, [256, 256, 1024])  
    X = identity_block(X, [256, 256, 1024])  

    # Stage 4
    X = projection_block(X, [512, 512, 2048], s=2)  
    X = identity_block(X, [512, 512, 2048])  
    X = identity_block(X, [512, 512, 2048])  

    # Average Pooling & Fully Connected Layer
    X = K.layers.AveragePooling2D(pool_size=(7, 7), strides=1)(X)
    X = K.layers.Flatten()(X)
    X = K.layers.Dense(units=1000, activation="softmax",
                       kernel_initializer=initializer)(X)

    # Create model
    model = K.models.Model(inputs=X_input, outputs=X)

    return model
