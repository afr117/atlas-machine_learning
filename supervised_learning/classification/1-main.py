#!/usr/bin/env python3


import numpy as np


# Import the Neuron class from the 1-neuron module
Neuron = __import__('1-neuron').Neuron

# Use dummy data to simulate feature matrix X
# Example: 4 input features with 5 samples
dummy_X = np.random.randn(4, 5)  # 4 features, 5 samples

# Initialize a Neuron instance with the number of features
neuron = Neuron(dummy_X.shape[0])

# Print private attributes using getter methods
print("Weights (W):")
print(neuron.W)
print("Bias (b):")
print(neuron.b)
print("Activation Output (A):")
print(neuron.A)

# Try setting the activation output to trigger the AttributeError
try:
    neuron.A = 10  # Trying to set the activation output (should raise AttributeError)
except AttributeError as e:
    print(f"Error: {e}")
