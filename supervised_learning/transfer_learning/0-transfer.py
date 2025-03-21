#!/usr/bin/env python3
"""
Trains a CNN on CIFAR-10 using transfer learning and MobileNetV2
"""
from tensorflow import keras as K
import numpy as np

def preprocess_data(X, Y):
    """
    Pre-process the data for the model
    """
    X_p = K.applications.mobilenet_v2.preprocess_input(
        K.layers.Resizing(96, 96)(X.astype('float32')))
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p.numpy(), Y_p

if __name__ == '__main__':
    # Load CIFAR-10
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # Load base model (frozen)
    base_model = K.applications.MobileNetV2(include_top=False,
                                            weights='imagenet',
                                            input_shape=(96, 96, 3),
                                            pooling='avg')
    base_model.trainable = False

    # Build model
    inputs = K.Input(shape=(96, 96, 3))
    x = base_model(inputs, training=False)
    outputs = K.layers.Dense(10, activation='softmax')(x)
    model = K.Model(inputs, outputs)

    # Compile
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train
    model.fit(X_train, Y_train,
              validation_data=(X_test, Y_test),
              epochs=10,
              batch_size=128,
              verbose=1)

    # Save
    model.save("cifar10.h5")
