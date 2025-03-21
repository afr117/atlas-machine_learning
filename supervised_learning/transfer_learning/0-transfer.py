#!/usr/bin/env python3
"""Trains a small CNN to classify CIFAR-10 and saves the model"""
from tensorflow import keras as K


def preprocess_data(X, Y):
    """
    Pre-processes the CIFAR-10 dataset
    - X: input images
    - Y: class labels
    Returns: (X_p, Y_p)
    """
    X = X.astype('float32') / 255.0
    Y = K.utils.to_categorical(Y, 10)
    return X, Y


if __name__ == '__main__':
    (X_train, Y_train), (X_valid, Y_valid) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_valid, Y_valid = preprocess_data(X_valid, Y_valid)

    model = K.models.Sequential([
        K.layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                        input_shape=(32, 32, 3)),
        K.layers.MaxPooling2D((2, 2)),
        K.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        K.layers.MaxPooling2D((2, 2)),
        K.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        K.layers.MaxPooling2D((2, 2)),
        K.layers.Flatten(),
        K.layers.Dense(256, activation='relu'),
        K.layers.Dropout(0.5),
        K.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              validation_data=(X_valid, Y_valid),
              epochs=15,
              batch_size=64,
              verbose=1)

    model.save('cifar10.h5')
