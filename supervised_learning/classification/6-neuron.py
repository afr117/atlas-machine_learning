#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Activation function: Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the Sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the model architecture and parameters
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases for input to hidden layer and hidden to output layer
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))
    
    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        
        return self.final_output
    
    def backward(self, X, y, learning_rate=0.1):
        # Calculate output layer error
        output_error = y - self.final_output
        output_delta = output_error * sigmoid_derivative(self.final_output)
        
        # Calculate hidden layer error
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases using gradient descent
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
    
    def train(self, X, y, epochs=10000, learning_rate=0.1):
        for epoch in range(epochs):
            # Forward and backward pass
            self.forward(X)
            self.backward(X, y, learning_rate)
            
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - self.final_output))
                print(f"Epoch {epoch}, Loss: {loss}")
    
    def predict(self, X):
        return self.forward(X)

# Load and prepare data
def load_data():
    # Example of generating dummy data for binary classification (e.g., XOR problem)
    X = np.random.randn(1000, 2)  # 1000 samples, 2 features
    y = np.array([[1] if x[0] + x[1] > 0 else [0] for x in X])  # Label based on sum of features
    
    return X, y

# Data preprocessing
def preprocess_data(X, y):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features (zero mean, unit variance)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Main function
if __name__ == "__main__":
    # Load the data
    X, y = load_data()
    
    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Initialize the neural network
    input_size = X_train.shape[1]  # Number of features
    hidden_size = 4  # Number of neurons in hidden layer
    output_size = 1  # Output layer (binary classification)
    
    nn = NeuralNetwork(input_size, hidden_size, output_size)
    
    # Train the neural network
    nn.train(X_train, y_train, epochs=10000, learning_rate=0.1)
    
    # Predict on the test set
    y_pred = nn.predict(X_test)
    y_pred = np.round(y_pred)  # Convert output to 0 or 1
    
    # Evaluate the model performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Plot results
    plt.figure(figsize=(10,6))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred.flatten(), cmap='coolwarm', marker='o', s=50, alpha=0.7)
    plt.title("Test Set Predictions")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar()
    plt.show()
