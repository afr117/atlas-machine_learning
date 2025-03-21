#!/usr/bin/env python3
"""
Trains a CNN on CIFAR-10 using transfer learning with MobileNetV2.
"""
from tensorflow import keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """
    Pre-processes CIFAR-10 data
    Args:
        X: numpy.ndarray (m, 32, 32, 3)
        Y: numpy.ndarray (m,)
    Returns:
        X, Y as preprocessed tf tensors
    """
    X = tf.cast(X, tf.float32)
    Y = tf.cast(Y, tf.int32)
    return X, Y


def prepare_dataset(X, Y, batch_size=32, training=True):
    """
    Builds a low-memory tf.data.Dataset for training or validation
    """
    X, Y = preprocess_data(X, Y)
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))

    def process(x, y):
        x = tf.image.resize(x, (96, 96))
        x = K.applications.mobilenet_v2.preprocess_input(x)
        return x, y

    dataset = dataset.map(process, num_parallel_calls=1)  # No parallel threads
    if training:
        dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    return dataset  # Removed prefetch and cache for low memory


if __name__ == '__main__':
    # Load CIFAR-10
    (X_train, Y_train), (X_val, Y_val) = K.datasets.cifar10.load_data()

    # Build datasets
    train_ds = prepare_dataset(X_train, Y_train, batch_size=16, training=True)
    val_ds = prepare_dataset(X_val, Y_val, batch_size=16, training=False)

    # Build MobileNetV2-based model
    base_model = K.applications.MobileNetV2(include_top=False,
                                            weights='imagenet',
                                            input_shape=(96, 96, 3),
                                            pooling='avg')
    base_model.trainable = False

    inputs = K.Input(shape=(96, 96, 3))
    x = base_model(inputs, training=False)
    x = K.layers.Dense(128, activation='relu')(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)
    model = K.Model(inputs, outputs)

    model.compile(optimizer=K.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train (reduce epochs for testing, increase when memory stable)
    model.fit(train_ds,
              validation_data=val_ds,
              epochs=3,
              verbose=1)

    # Save model
    model.save('cifar10.h5')
