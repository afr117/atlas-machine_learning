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
    X = tf.cast(X, tf.float32)
    Y = tf.cast(Y, tf.int32)
    return X, Y

def prepare_dataset(X, Y, batch_size=64, training=True):
    """
    Builds an efficient tf.data.Dataset
    """
    X, Y = preprocess_data(X, Y)
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))

    def process(x, y):
        x = tf.image.resize(x, (96, 96))
        x = K.applications.mobilenet_v2.preprocess_input(x)
        y = tf.one_hot(y, 10)
        y = tf.reshape(y, [-1])
        return x, y

    dataset = dataset.map(process, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(1000)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

if __name__ == '__main__':
    # Load and prepare CIFAR-10
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    train_ds = prepare_dataset(X_train, Y_train, training=True)
    val_ds = prepare_dataset(X_test, Y_test, training=False)

    # Build model using MobileNetV2
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
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=5,
              verbose=1)

    model.save('cifar10.h5')
