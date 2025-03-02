#!/usr/bin/env python3

import tensorflow as tf

def save_model(network, filename):
    """
    Saves an entire model to a file.
    :param network: the model to save
    :param filename: the path of the file that the model should be saved to
    """
    network.save(filename)

def load_model(filename):
    """
    Loads an entire model from a file.
    :param filename: the path of the file that the model should be loaded from
    :return: the loaded model
    """
    return tf.keras.models.load_model(filename)
