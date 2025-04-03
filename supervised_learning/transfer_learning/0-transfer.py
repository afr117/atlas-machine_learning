#!/usr/bin/env python3
"""
Transfer learning on CIFAR-10 using EfficientNetB0 (memory-friendly)
"""
from tensorflow import keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """
    Normalize images and one-hot encode labels
    """
    X = X.astype('float32') / 255.0
    Y = K.utils.to_categorical(Y, 10)
    return X, Y


def resize_and_preprocess(image, label):
    """
    Resize image to 64x64 and preprocess for EfficientNet
    """
    image = tf.image.resize(image, (64, 64))
    image = K.applications.efficientnet.preprocess_input(image)
    return image, label


if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    batch_size = 32
    AUTOTUNE = tf.data.AUTOTUNE

    # Light augmentation
    data_aug = K.Sequential([K.layers.RandomFlip("horizontal")])

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_ds = train_ds.map(resize_and_preprocess, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(lambda x, y: (data_aug(x), y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.shuffle(500).batch(batch_size).prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    val_ds = val_ds.map(resize_and_preprocess, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

    base = K.applications.EfficientNetB0(
        input_shape=(64, 64, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )
    base.trainable = False

    inputs = K.Input(shape=(64, 64, 3))
    x = base(inputs, training=False)
    x = K.layers.Dense(128, activation='relu')(x)
    x = K.layers.Dropout(0.3)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=10)

    model.save('cifar10.h5')
