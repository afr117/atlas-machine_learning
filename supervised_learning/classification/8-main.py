#!/usr/bin/env python3

import numpy as np

# Import the NeuralNetwork class
NeuralNetwork = __import__('8-neural_network').NeuralNetwork

# Load data
lib_train = np.load('data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

# Initialize a neural network with input size (features) and 3 nodes in hidden layer
np.random.seed(0)
nn = NeuralNetwork(X.shape[0], 3)

# Print weights and biases for verification
print(nn.W1)
print(nn.W1.shape)
print(nn.b1)
print(nn.W2)
print(nn.W2.shape)
print(nn.b2)
print(nn.A1)
print(nn.A2)

# Test attribute assignment for A1
nn.A1 = 10
print(nn.A1)
