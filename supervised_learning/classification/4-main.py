#!/usr/bin/env python3

import numpy as np
import os

# Import the Neuron class
Neuron = __import__('4-neuron').Neuron

# Check if the file exists
data_path = '../data/Binary_Train.npz'
if os.path.isfile(data_path):
    lib_train = np.load(data_path)
    X_3D, Y = lib_train['X'], lib_train['Y']
    X = X_3D.reshape((X_3D.shape[0], -1)).T
else:
    print(f"File not found: {data_path}")
    # Create dummy data for testing
    X = np.random.randint(0, 2, (10, 100))  # 10 features, 100 examples
    Y = np.random.randint(0, 2, (1, 100))   # 1 label row, 100 examples

# Initialize and evaluate the neuron
np.random.seed(0)
neuron = Neuron(X.shape[0])
A, cost = neuron.evaluate(X, Y)

print(A)
print(cost)
