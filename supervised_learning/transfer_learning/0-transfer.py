#!/usr/bin/env python3
"""
Transfer learning with MobileNetV2 on a small CIFAR-10 subset.
Optimized for very low memory by using 32x32 input size.
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
    image = tf.image.resize(image, (32, 32))  # downsized for memory
    image = K.applications.mobilenet_v2.preprocess_input(image)
    return image, label


if __name__ == '__main__':
    # Load and preprocess smaller subset (10k train / 2k test)
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

    # Load MobileNetV2 base with smaller input
    base_model = K.applications.MobileNetV2(
        input_shape=(32, 32, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    # Fine-tune top half
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) // 2
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Data augmentation
    data_augmentation = K.Sequential([
        K.layers.RandomFlip("horizontal"),
        K.layers.RandomRotation(0.1),
        K.layers.RandomZoom(0.1),
    ])

    # Build model
    inputs = K.Input(shape=(32, 32, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=True)
    x = K.layers.Dense(128, activation='relu')(x)
    x = K.layers.Dropout(0.3)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.Model(inputs, outputs)

    # Compile
    model.compile(
        optimizer=K.optimizers.RMSprop(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    model.fit(train_ds, validation_data=val_ds, epochs=30)

    # Save
    model.save('cifar10.h5')
