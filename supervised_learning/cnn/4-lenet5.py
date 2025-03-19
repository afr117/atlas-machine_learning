#!/usr/bin/env python3
"""
Builds a modified version of the LeNet-5 architecture using TensorFlow v1.
"""
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


def lenet5(x, y):
    """
    Builds a modified LeNet-5 architecture.

    Parameters:
    - x: tf.placeholder of shape (m, 28, 28, 1) containing input images
    - y: tf.placeholder of shape (m, 10) containing one-hot labels

    Returns:
    - y_pred: tensor for the softmax activated output
    - train_op: training operation utilizing Adam optimization
    - loss: tensor for the loss of the network
    - accuracy: tensor for the accuracy of the network
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    # First convolutional layer: (5x5 kernel, 6 filters, same padding)
    conv1 = tf.layers.conv2d(x, filters=6, kernel_size=5, padding="same",
                             activation=tf.nn.relu, kernel_initializer=initializer)

    # Max pooling layer (2x2 kernel, stride 2x2)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

    # Second convolutional layer: (5x5 kernel, 16 filters, valid padding)
    conv2 = tf.layers.conv2d(pool1, filters=16, kernel_size=5, padding="valid",
                             activation=tf.nn.relu, kernel_initializer=initializer)

    # Max pooling layer (2x2 kernel, stride 2x2)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)

    # Flatten the pooled output
    flatten = tf.layers.flatten(pool2)

    # Fully connected layer with 120 nodes
    fc1 = tf.layers.dense(flatten, units=120, activation=tf.nn.relu,
                          kernel_initializer=initializer)

    # Fully connected layer with 84 nodes
    fc2 = tf.layers.dense(fc1, units=84, activation=tf.nn.relu,
                          kernel_initializer=initializer)

    # Output layer (logits, no activation yet)
    y_logits = tf.layers.dense(fc2, units=10, kernel_initializer=initializer)

    # Softmax activation
    y_pred = tf.nn.softmax(y_logits)

    # Loss computation (cross-entropy using logits, not softmax output)
    loss = tf.losses.softmax_cross_entropy(y, y_logits)

    # Accuracy calculation
    correct_predictions = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # Training operation using Adam optimizer
    train_op = tf.train.AdamOptimizer().minimize(loss)

    return y_pred, train_op, loss, accuracy
