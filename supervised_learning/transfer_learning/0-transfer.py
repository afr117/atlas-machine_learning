#!/usr/bin/env python3
"""
Trains a CNN on CIFAR-10 using transfer learning with MobileNetV2
"""
from tensorflow import keras as K
import tensorflow as tf

def preprocess_data(X, Y):
    """
    Pre-processes the data for the model
    """
    X = tf.image.resize(X, (96, 96))  # Resizes on the fly
    X = K.applications.mobilenet_v2.preprocess_input(X)
    Y = K.utils.to_categorical(Y, 10)
    return X, Y

if __name__ == '__main__':
    # Load and preprocess data
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # Build model with MobileNetV2
    base_model = K.applications.MobileNetV2(include_top=False,
                                            weights='imagenet',
                                            input_shape=(96, 96, 3),
                                            pooling='avg')
    base_model.trainable = False

    inputs = K.Input(shape=(96, 96, 3))
    x = base_model(inputs, training=False)
    x = K.layers.Dense(256, activation='relu')(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)
    model = K.Model(inputs, outputs)

    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              epochs=5,
              batch_size=64,
              validation_data=(X_test, Y_test),
              verbose=1)

    model.save('cifar10.h5')
