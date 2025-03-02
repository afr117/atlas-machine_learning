#!/usr/bin/env python3
"""
Module for saving and loading a Keras model.
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model to a file.

    Parameters:
    network (K.Model): The model to save.
    filename (str): The path of the file where the model should be saved.

    Returns:
    None
    """
    network.save(filename)


def load_model(filename):
    """
    Loads an entire model from a file.

    Parameters:
    filename (str): The path of the file where the model should be loaded from.

    Returns:
    K.Model: The loaded model.
    """
    return K.models.load_model(filename)
