#!/usr/bin/env python3

import numpy as np

class Neuron:
    def __init__(self, nx):
        # Ensure nx is an integer
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        
        # Ensure nx is a positive integer
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        # Initialize weights W with a random normal distribution
        self.W = np.random.randn(1, nx)
        
        # Initialize bias b to 0
        self.b = 0
        
        # Initialize activation output A to 0
        self.A = 0
