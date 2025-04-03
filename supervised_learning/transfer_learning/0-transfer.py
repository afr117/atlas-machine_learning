#!/usr/bin/env python3
"""
Improved transfer learning with MobileNetV2 on small CIFAR-10 subset.
"""
from tensorflow import keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """
    Normalize and one-hot encode
    """
    X = X.astype('float32') / 255.0
    Y = K.utils.to_categorical(Y, 10)
    return X, Y


def preprocess_and_resize(image, label):
    """
    Resize and preprocess image
    """
    image = tf.image.resize(image, (64, 64))
    image = K.applications.mobilenet_v2.preprocess_input(image)
    return image, label


if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = X_train[:10000], Y_train[:10000]
    X_test, Y_test = X_test[:2000], Y_test[:2000]

    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    batch_size = 32

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_ds = train_ds.map(preprocess_and_resize).shuffle(1000).batch(batch_size).prefetch(1)

    val_ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    val_ds = val_ds.map(preprocess_and_resize).batch(batch_size).prefetch(1)

    base_model = K.applications.MobileNetV2(
        input_shape=(64, 64, 3),
        include_top=False,
        weights='imagenet',
        alpha=0.35,  # reduce base model size
    )
    base_model.trainable = False

    inputs = K.Input(shape=(64, 64, 3))
    x = base_model(inputs, training=False)
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dense(128, activation='relu')(x)
    x = K.layers.Dropout(0.3)(x)
    x = K.layers.Dense(64, activation='relu')(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.Model(inputs, outputs)
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Learning rate scheduler
    callback = K.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1,
        min_lr=1e-5
    )

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=10,
              callbacks=[callback])

    model.save('cifar10.h5')
