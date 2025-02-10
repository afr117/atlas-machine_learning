#!/usr/bin/env python3
"""Evaluates the output of a trained neural network model."""

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network.

    Args:
        X (np.ndarray): Input data for evaluation.
        Y (np.ndarray): One-hot labels for X.
        save_path (str): Path to the saved model.

    Returns:
        np.ndarray: Network's prediction (one-hot encoded).
        float: Accuracy of the network.
        float: Loss of the network.
    """
    with tf.Session() as sess:
        # Load the saved model
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        # Retrieve necessary tensors from the graph's collections
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        loss = tf.get_collection("loss")[0]
        accuracy = tf.get_collection("accuracy")[0]

        # Run the session to get predictions, accuracy, and loss
        y_pred_eval, acc_eval, loss_eval = sess.run(
            [y_pred, accuracy, loss], feed_dict={x: X, y: Y}
        )

    return y_pred_eval, acc_eval, loss_eval
