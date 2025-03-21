#!/usr/bin/env python3
"""Trains a CNN using transfer learning to classify CIFAR-10"""
from tensorflow import keras as K


def preprocess_data(X, Y):
    """Pre-processes CIFAR-10 data"""
    X = X.astype('float32') / 255.0
    Y = K.utils.to_categorical(Y, 10)
    return X, Y


if __name__ == '__main__':
    (X_train, Y_train), (X_valid, Y_valid) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_valid, Y_valid = preprocess_data(X_valid, Y_valid)

    # Use MobileNetV2 structure but no pretrained weights to avoid download
    base_model = K.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights=None
    )

    base_model.trainable = True

    inputs = K.Input(shape=(32, 32, 3))
    resize = K.layers.Resizing(96, 96)(inputs)
    base_output = base_model(resize)
    avg_pool = K.layers.GlobalAveragePooling2D()(base_output)
    dropout = K.layers.Dropout(0.2)(avg_pool)
    outputs = K.layers.Dense(10, activation='softmax')(dropout)

    model = K.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              validation_data=(X_valid, Y_valid),
              epochs=15,
              batch_size=64,
              verbose=1)

    model.save('cifar10.h5')
