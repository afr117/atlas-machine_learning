#!/usr/bin/env python3

import numpy as np
import os  # Import os for file operations
import zipfile  # Import zipfile to handle ZIP files
from import_neuron import Neuron  # Import Neuron using the wrapper script

# Define the path to the ZIP file and the target extraction path
zip_file_path = 'data/Binary_Train.zip'
extract_to = 'data/'

# Check if the ZIP file exists
if not os.path.exists(zip_file_path):
    raise FileNotFoundError(f"The ZIP file was not found at the path: {zip_file_path}")

# Extract the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

# Now you can load the extracted NPZ file
npz_file_path = os.path.join(extract_to, 'Binary_Train.npz')
if not os.path.exists(npz_file_path):
    raise FileNotFoundError(f"The NPZ file was not found at the path: {npz_file_path}")

lib_train = np.load(npz_file_path)
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
A, cost = neuron.evaluate(X, Y)
print(A)
print(cost)
