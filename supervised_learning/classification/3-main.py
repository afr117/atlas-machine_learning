#!/usr/bin/env python3
"""
Test script for the Neuron class, testing the cost computation functionality.
"""

import numpy as np
import importlib.util

# Define the path to the module
module_path = './3-neuron.py'

# Load the module using importlib
spec = importlib.util.spec_from_file_location("Neuron", module_path)
neuron_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(neuron_module)

# Access the Neuron class from the dynamically loaded module
Neuron = neuron_module.Neuron

# Generate dummy data to avoid file not found errors
np.random.seed(0)  # For reproducibility
X = np.random.randn(3, 5)  # 3 features and 5 examples
Y = np.random.randint(0, 2, (1, 5))  # Binary labels

# Initialize the neuron
neuron = Neuron(X.shape[0])

# Perform forward propagation
A = neuron.forward_prop(X)

# Calculate the cost
cost = neuron.cost(Y, A)

# Print the cost
print(cost)
