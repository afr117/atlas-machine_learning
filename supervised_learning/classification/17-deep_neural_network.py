#!/usr/bin/env python3
"""
This module defines a Deep Neural Network (DNN)
class for performing binary classification.
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification.

    Attributes:
    - __L (int): The number of layers in the network.
    - __cache (dict): A dictionary to store intermediary values of the network.
    - __weights (dict): A dictionary to store the
    weights and biases of the network.
    """

    def __init__(self, nx, layers):
        """
        Initializes a deep neural network.

        Parameters:
        - nx (int): Number of input features.
        - layers (list): List containing the number of nodes in each layer.

        Raises:
        - TypeError: If nx is not an integer.
        - ValueError: If nx is less than 1.
        - TypeError: If layers is not a list of positive integers.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if any(map(lambda n: not isinstance(n, int) or n <= 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        def initialize_weights(index, prev_layer):
            """Recursively initializes the weights and
            biases for each layer."""
            if index > self.__L:
                return
            self.__weights[f"W{index}"] = (
                np.random.randn(layers[index - 1], prev_layer) *
                np.sqrt(2 / prev_layer)
            )
            self.__weights[f"b{index}"] = np.zeros((layers[index - 1], 1))
            initialize_weights(index + 1, layers[index - 1])

        initialize_weights(1, nx)

    @property
    def L(self):
        """Returns the number of layers in the neural network."""
        return self.__L

    @property
    def cache(self):
        """Returns the cache dictionary storing intermediary values."""
        return self.__cache

    @property
    def weights(self):
        """Returns the weights dictionary storing weights and biases."""
        return self.__weights
