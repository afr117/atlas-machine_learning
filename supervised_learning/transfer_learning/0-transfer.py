#!/usr/bin/env python3
"""Trains a tiny CNN to classify CIFAR-10 and saves the model"""
from tensorflow import keras as K


def preprocess_data(X, Y):
    """Pre-process CIFAR-10 data"""
    X = X.astype('float32') / 255.0
    Y = K.utils.to_categorical(Y, 10)
    return X, Y


if __name__ == '__main__':
    (X_train, Y_train), (X_val, Y_val) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_val, Y_val = preprocess_data(X_val, Y_val)

    model = K.models.Sequential([
        K.layers.Input(shape=(32, 32, 3)),
        K.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        K.layers.MaxPooling2D(),
        K.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        K.layers.MaxPooling2D(),
        K.layers.Flatten(),
        K.layers.Dense(64, activation='relu'),
        K.layers.Dropout(0.3),
        K.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              epochs=5,
              batch_size=128,
              validation_data=(X_val, Y_val),
              verbose=1)

    model.save('cifar10.h5')
