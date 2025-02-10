#!/usr/bin/env python3
"""
Defines a single neuron performing binary classification
"""
import numpy as np


class Neuron:
    """
    Class that defines a single neuron for binary classification
    """
    def __init__(self, nx):
        """
        Initializes the neuron
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        np.random.seed(0)  # Ensure deterministic initialization
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0
    
    @property
    def W(self):
        return self.__W
    
    @property
    def b(self):
        return self.__b
    
    @property
    def A(self):
        return self.__A
