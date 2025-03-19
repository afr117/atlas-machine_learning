#!/usr/bin/env python3
"""
Builds a modified version of the LeNet-5 architecture using TensorFlow 1.x.
"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture.

    Parameters:
    x (tf.placeholder): Input images, shape (m, 28, 28, 1).
    y (tf.placeholder): One-hot labels, shape (m, 10).

    Returns:
    y_pred (tensor): Softmax activated output tensor.
    train_op (tensor): Training operation using Adam optimizer.
    loss (tensor): Loss of the network.
    acc (tensor): Accuracy of the network.
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    # **First Convolutional Layer**: 6 filters, 5x5 kernel, 'same' padding
    conv1 = tf.layers.conv2d(x, filters=6, kernel_size=5, padding='same',
                             activation=tf.nn.relu, kernel_initializer=initializer)

    # **Max Pooling Layer**: 2x2 kernel, 2x2 stride
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

    # **Second Convolutional Layer**: 16 filters, 5x5 kernel, 'valid' padding
    conv2 = tf.layers.conv2d(pool1, filters=16, kernel_size=5, padding='valid',
                             activation=tf.nn.relu, kernel_initializer=initializer)

    # **Max Pooling Layer**: 2x2 kernel, 2x2 stride
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)

    # Flatten the output for fully connected layers
    flat = tf.layers.flatten(pool2)

    # **Fully Connected Layer**: 120 nodes, ReLU activation
    fc1 = tf.layers.dense(flat, units=120, activation=tf.nn.relu,
                          kernel_initializer=initializer)

    # **Fully Connected Layer**: 84 nodes, ReLU activation
    fc2 = tf.layers.dense(fc1, units=84, activation=tf.nn.relu,
                          kernel_initializer=initializer)

    # **Fully Connected Output Layer**: 10 nodes (softmax activation)
    y_pred = tf.layers.dense(fc2, units=10, activation=tf.nn.softmax,
                             kernel_initializer=initializer)

    # **Loss Function**: Categorical cross-entropy
    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    # **Accuracy Calculation**
    correct_prediction = tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(y, axis=1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # **Optimization**: Adam optimizer with default parameters
    train_op = tf.train.AdamOptimizer().minimize(loss)

    return y_pred, train_op, loss, acc
