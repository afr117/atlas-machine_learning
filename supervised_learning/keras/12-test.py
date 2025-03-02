#!/usr/bin/env python3
"""
Module for testing a trained neural network in Keras.
"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network model.

    Parameters:
    network (K.Model): The trained model to test.
    data (numpy.ndarray): The input data to test the model with.
    labels (numpy.ndarray): The correct one-hot labels of the data.
    verbose (bool): Determines if output should be
    printed during the testing process.

    Returns:
    tuple: The loss and accuracy of the model with
    the testing data, respectively.
    """
    return network.evaluate(data, labels, verbose=verbose)
