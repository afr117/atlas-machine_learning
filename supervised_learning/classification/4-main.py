#!/usr/bin/env python3

import numpy as np
import importlib.util

# Dynamically load the 4-neuron.py module
module_name = '4-neuron'
module_path = './4-neuron.py'

spec = importlib.util.spec_from_file_location(module_name, module_path)
neuron_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(neuron_module)

# Access the Neuron class
Neuron = neuron_module.Neuron

# Simulate a dataset if the actual file is not available
# Define the number of input features and examples
nx = 5  # Number of input features
m = 10  # Number of examples

# Generate random data
np.random.seed(0)
X = np.random.randn(nx, m)
Y = np.random.randint(0, 2, (1, m))

# Initialize the neuron with the number of input features
neuron = Neuron(nx)

# Evaluate the neuron
A, cost = neuron.evaluate(X, Y)

# Print the outputs
print("Predictions:")
print(A)
print("Cost:")
print(cost)
