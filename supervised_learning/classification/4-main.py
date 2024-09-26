#!/usr/bin/env python3

import numpy as np

# Importing the Neuron class from the 4-neuron.py file
Neuron = __import__('4-neuron').Neuron

# Loading the training data from the Binary_Train.npz file in the data directory
lib_train = np.load('data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']

# Reshaping the input data to the required format
X = X_3D.reshape((X_3D.shape[0], -1)).T

# Setting the random seed for reproducibility
np.random.seed(0)

# Initializing the Neuron instance
neuron = Neuron(X.shape[0])

# Evaluating the neuron's predictions and calculating the cost
A, cost = neuron.evaluate(X, Y)

# Printing the predictions and the cost, rounding the cost to 6 decimal places
print(A)
print(round(cost, 6))
