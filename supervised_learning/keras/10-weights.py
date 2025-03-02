#!/usr/bin/env python3
"""
Module for saving and loading model weights in Keras.
"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Saves a model’s weights to a file.

    Parameters:
    network (K.Model): The model whose weights should be saved.
    filename (str): The path of the file where the weights should be saved.
    save_format (str): The format in which the weights should be saved
                       (default is 'keras').

    Returns:
    None
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
    Loads a model’s weights from a file.

    Parameters:
    network (K.Model): The model to which the weights should be loaded.
    filename (str): The path of the file from where the
    weights should be loaded.

    Returns:
    None
    """
    network.load_weights(filename)
