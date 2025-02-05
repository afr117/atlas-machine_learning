#!/usr/bin/env python3
import numpy as np

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
            # Forward pass
            self.forward(X)
            
            # Backward pass
            self.backward(X, y, learning_rate)
            
            # Print loss every 1000 epochs
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

# Split the dataset manually (train-test split)
def train_test_split(X, y, test_size=0.2):
    num_samples = X.shape[0]
    num_train = int((1 - test_size) * num_samples)
    
    X_train, X_test = X[:num_train], X[num_train:]
    y_train, y_test = y[:num_train], y[num_train:]
    
    return X_train, X_test, y_train, y_test

# Standardization: Zero mean, unit variance
def standardize(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    return X_train, X_test

# Main function
if __name__ == "__main__":
    # Load the data
    X, y = load_data()
    
    # Preprocess the data (train-test split and standardization)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_test = standardize(X_train, X_test)
    
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
    
    # Evaluate the model performance (accuracy)
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Confusion Matrix (Manual computation)
    tp = np.sum((y_pred == 1) & (y_test == 1))  # True Positives
    tn = np.sum((y_pred == 0) & (y_test == 0))  # True Negatives
    fp = np.sum((y_pred == 1) & (y_test == 0))  # False Positives
    fn = np.sum((y_pred == 0) & (y_test == 1))  # False Negatives
    
    print("Confusion Matrix:")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
