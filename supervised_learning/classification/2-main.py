#!/usr/bin/env python3
"""
Test script for the Neuron class, testing the forward propagation functionality.
"""

import numpy as np
import importlib.util

# Define the path to the module
module_path = './2-neuron.py'

# Load the module using importlib
spec = importlib.util.spec_from_file_location("Neuron", module_path)
neuron_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(neuron_module)

# Access the Neuron class from the dynamically loaded module
Neuron = neuron_module.Neuron

# Generate dummy data
np.random.seed(0)  # For reproducibility
X = np.random.randn(10, 100)  # 10 features and 100 examples
Y = np.random.randint(0, 2, (1, 100))  # Binary labels

# Initialize the neuron
neuron = Neuron(X.shape[0])

# Set bias to 1 (as per the example)
neuron._Neuron__b = 1

# Perform forward propagation
A = neuron.forward_prop(X)

# Print the activated output if the object reference matches
if A is neuron.A:
    print(A)
