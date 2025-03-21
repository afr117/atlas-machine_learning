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
    X = X.astype('float32')
    X_resized = np.zeros((X.shape[0], 96, 96, 3), dtype='float32')
    resize_layer = K.layers.Resizing(96, 96)
    for i in range(0, X.shape[0], 1000):  # process in batches
        X_resized[i:i+1000] = resize_layer(X[i:i+1000])
    X_p = K.applications.mobilenet_v2.preprocess_input(X_resized)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

if __name__ == '__main__':
    # Load CIFAR-10
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # Base model
    base = K.applications.MobileNetV2(include_top=False,
                                      input_shape=(96, 96, 3),
                                      pooling='avg',
                                      weights='imagenet')
    base.trainable = False

    # Custom head
    inputs = K.Input(shape=(96, 96, 3))
    x = base(inputs, training=False)
    x = K.layers.Dense(256, activation='relu')(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)
    model = K.Model(inputs, outputs)

    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              epochs=5,
              batch_size=64,  # smaller batch size = lower memory usage
              validation_data=(X_test, Y_test),
              verbose=1)

    model.save('cifar10.h5')
