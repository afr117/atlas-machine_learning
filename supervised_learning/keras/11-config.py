#!/usr/bin/env python3
"""
Module for saving and loading model configurations in Keras.
"""

import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model’s configuration in
    JSON format.

    Parameters:
    network (K.Model): The model whose
    configuration should be saved.
    filename (str): The path of the file where the
    configuration should be saved.

    Returns:
    None
    """
    config = network.to_json()
    with open(filename, "w") as f:
        f.write(config)


def load_config(filename):
    """
    Loads a model with a specific configuration from
    a JSON file.

    Parameters:
    filename (str): The path of the file containing the
    model’s configuration in JSON format.

    Returns:
    K.Model: The loaded model with the specified configuration.
    """
    with open(filename, "r") as f:
        config = f.read()

    return K.models.model_from_json(config)
