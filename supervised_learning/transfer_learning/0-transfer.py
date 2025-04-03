#!/usr/bin/env python3
"""
Transfer learning on CIFAR-10 using MobileNetV2 with data augmentation and fine-tuning.
"""
from tensorflow import keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """
    Normalize images and convert labels to one-hot
    """
    X = X.astype('float32') / 255.0
    Y = K.utils.to_categorical(Y, 10)
    return X, Y


def preprocess_and_resize(image, label):
    """
    Resize and preprocess image for MobileNetV2
    """
    image = tf.image.resize(image, (96, 96))
    image = K.applications.mobilenet_v2.preprocess_input(image)
    return image, label


if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    batch_size = 64
    AUTOTUNE = tf.data.AUTOTUNE

    # Data Augmentation
    data_augmentation = K.Sequential([
        K.layers.RandomFlip("horizontal"),
        K.layers.RandomRotation(0.1),
        K.layers.RandomZoom(0.1),
    ])

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_ds = train_ds.shuffle(1000).map(
        lambda x, y: preprocess_and_resize(x, y), num_parallel_calls=AUTOTUNE
    )
    train_ds = train_ds.batch(batch_size).map(
        lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    val_ds = val_ds.map(preprocess_and_resize, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

    # Base Model
    base_model = K.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False  # Freeze initially

    inputs = K.Input(shape=(96, 96, 3))
    x = base_model(inputs, training=False)
    x = K.layers.Dense(128, activation='relu')(x)
    x = K.layers.Dropout(0.3)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # First Training Phase
    early_stop = K.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=10,
              callbacks=[early_stop])

    # Fine-tune (unfreeze top MobileNetV2 layers)
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(optimizer=K.optimizers.Adam(1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=5,
              callbacks=[early_stop])

    model.save('cifar10.h5')
