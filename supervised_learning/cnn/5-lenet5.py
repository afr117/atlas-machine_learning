#!/usr/bin/env python3
"""
Builds a modified version of the LeNet-5 architecture using Keras.
"""
from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified LeNet-5 architecture.

    Parameters:
    - X: K.Input of shape (m, 28, 28, 1) containing input images

    Returns:
    - A compiled K.Model using Adam optimizer and accuracy metric.
    """
    initializer = K.initializers.he_normal(seed=0)

    # First Convolutional Layer: 6 filters, 5x5 kernel, same padding
    conv1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5), padding="same",
                            activation="relu", kernel_initializer=initializer)(X)

    # Max Pooling Layer: 2x2 kernel, stride 2x2
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    # Second Convolutional Layer: 16 filters, 5x5 kernel, valid padding
    conv2 = K.layers.Conv2D(filters=16, kernel_size=(5, 5), padding="valid",
                            activation="relu", kernel_initializer=initializer)(pool1)

    # Max Pooling Layer: 2x2 kernel, stride 2x2
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    # Flatten layer
    flatten = K.layers.Flatten()(pool2)

    # Fully Connected Layer with 120 nodes
    fc1 = K.layers.Dense(units=120, activation="relu", kernel_initializer=initializer)(flatten)

    # Fully Connected Layer with 84 nodes
    fc2 = K.layers.Dense(units=84, activation="relu", kernel_initializer=initializer)(fc1)

    # Output Layer with 10 nodes and softmax activation
    output = K.layers.Dense(units=10, activation="softmax", kernel_initializer=initializer)(fc2)

    # Create model
    model = K.Model(inputs=X, outputs=output)

    # Compile the model using Adam optimizer and categorical cross-entropy loss
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model
