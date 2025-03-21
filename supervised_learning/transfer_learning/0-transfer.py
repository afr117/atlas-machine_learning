#!/usr/bin/env python3
"""
Trains a CNN with transfer learning on CIFAR-10 and saves the model.
"""

from tensorflow import keras as K
from tensorflow.keras import layers
import numpy as np


def preprocess_data(X, Y):
    """
    Preprocesses the CIFAR-10 data.

    Args:
        X: np.ndarray - shape (m, 32, 32, 3), image data.
        Y: np.ndarray - shape (m, ), labels.

    Returns:
        X_p: preprocessed image data
        Y_p: one-hot encoded labels
    """
    X_p = X.astype('float32') / 255.0
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == "__main__":
    # Load and preprocess data
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # Resize images to match input size of pretrained model
    input_tensor = K.Input(shape=(32, 32, 3))
    resize = layers.Lambda(lambda image: K.backend.resize_images(image, 7, 7, "channels_last"))(input_tensor)

    # Load base model
    base_model = K.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_tensor=resize,
        pooling='avg'
    )
    base_model.trainable = False  # Freeze base model

    # Add classification head
    output = layers.Dense(10, activation='softmax')(base_model.output)
    model = K.Model(inputs=input_tensor, outputs=output)

    # Compile and train
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size=128,
              epochs=10,
              validation_data=(X_test, Y_test),
              verbose=1)

    # Save model
    model.save("cifar10.h5")
