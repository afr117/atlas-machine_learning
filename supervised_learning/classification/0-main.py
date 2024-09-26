#!/usr/bin/env python3


import numpy as np
Neuron = __import__('0-neuron').Neuron

# Simulate the input data shape (e.g., 784 for a 28x28 image in a flattened form)
X_shape = 784

# Set random seed for reproducibility
np.random.seed(0)

# Instantiate a Neuron object with X_shape inputs
neuron = Neuron(X_shape)

# Print the weights, shape, bias, and activated output
print("Weights (W):", neuron.W)
print("Weights shape:", neuron.W.shape)
print("Bias (b):", neuron.b)
print("Activated output (A):", neuron.A)

# Modify and print the activated output
neuron.A = 10
print("Updated activated output (A):", neuron.A)
