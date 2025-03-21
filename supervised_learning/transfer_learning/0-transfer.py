#!/usr/bin/env python3
"""
Transfer learning with MobileNetV2 on CIFAR-10
"""

from tensorflow import keras as K
import numpy as np

def preprocess_data(X, Y):
    """
    Preprocess the data by normalizing and one-hot encoding
    """
    X = X.astype('float32') / 255.0
    Y = K.utils.to_categorical(Y, 10)
    return X, Y


if __name__ == '__main__':
    # Load CIFAR-10 dataset
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # Resize on-the-fly using ImageDataGenerator
    datagen = K.preprocessing.image.ImageDataGenerator(
        preprocessing_function=K.applications.mobilenet_v2.preprocess_input,
        validation_split=0.1
    )

    train_gen = datagen.flow(
        X_train, Y_train, batch_size=64, subset='training', shuffle=True,
        target_size=(96, 96)
    )
    val_gen = datagen.flow(
        X_train, Y_train, batch_size=64, subset='validation', shuffle=False,
        target_size=(96, 96)
    )

    base_model = K.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False

    model = K.models.Sequential([
        base_model,
        K.layers.Dense(128, activation='relu'),
        K.layers.Dropout(0.3),
        K.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=5,
        verbose=1
    )

    model.save('cifar10.h5')
