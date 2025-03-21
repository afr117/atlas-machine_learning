#!/usr/bin/env python3
"""
Transfer learning with MobileNetV2 on CIFAR-10
"""
from tensorflow import keras as K
import numpy as np


def preprocess_data(X, Y):
    """
    Normalizes the data and converts labels to one-hot
    """
    X = X.astype('float32') / 255.0
    Y = K.utils.to_categorical(Y, 10)
    return X, Y


if __name__ == '__main__':
    # Load and preprocess data
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    base_model = K.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False

    inputs = K.Input(shape=(32, 32, 3))
    resize = K.layers.Lambda(lambda image: K.backend.resize_images(image, height_factor=3, width_factor=3,
                                                                    data_format='channels_last', interpolation='bilinear'))(inputs)
    preprocessed = K.applications.mobilenet_v2.preprocess_input(resize)

    x = base_model(preprocessed, training=False)
    x = K.layers.Dense(128, activation='relu')(x)
    x = K.layers.Dropout(0.3)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              validation_split=0.1,
              epochs=5,
              batch_size=64,
              verbose=1)

    model.save('cifar10.h5')
