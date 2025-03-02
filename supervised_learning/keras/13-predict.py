#!/usr/bin/env python3
"""
Module for making predictions using a
trained neural network in Keras.
"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a trained
    neural network model.

    Parameters:
    network (K.Model): The trained model to make
    the prediction with.
    data (numpy.ndarray): The input data to make
    the prediction with.
    verbose (bool): Determines if output should be
    printed during the prediction process.

    Returns:
    numpy.ndarray: The predicted output for the input data.
    """
    return network.predict(data, verbose=verbose)
