# Optimization Techniques in Deep Learning

## Overview
This project implements various optimization techniques used in deep learning, including feature scaling, batch normalization, gradient descent variations, and adaptive learning rate methods. Each implementation adheres to TensorFlow and NumPy standards, ensuring efficient computation and compatibility with machine learning workflows.

## Implemented Methods

### 1. **Feature Scaling**
- Implements standardization and normalization techniques.
- Enhances model convergence speed and stability.

### 2. **Batch Normalization**
- Standardizes activations to improve learning stability.
- Reduces internal covariate shift.

### 3. **Gradient Descent Variants**
- **Mini-batch Gradient Descent**: Balances computational efficiency and convergence.
- **Momentum-Based Gradient Descent**: Accelerates convergence using past gradients.
- **RMSProp Optimization**: Adapts learning rates based on gradient magnitudes.
- **Adam Optimization**: Combines momentum and adaptive learning rates.

### 4. **Learning Rate Scheduling**
- **Inverse Time Decay**: Reduces learning rate over epochs to refine model training.
- **TensorFlow-based Learning Rate Decay**: Implements built-in scheduling methods.

## Files and Implementations

- `0-norm_constants.py`: Computes normalization constants for feature scaling.
- `1-normalize.py`: Standardizes data using mean and standard deviation.
- `2-shuffle_data.py`: Shuffles dataset to enhance model generalization.
- `3-mini_batch.py`: Implements mini-batch gradient descent.
- `4-moving_average.py`: Computes moving averages for optimization.
- `5-momentum.py`: Implements gradient descent with momentum.
- `6-momentum.py`: TensorFlow-based momentum optimizer.
- `7-RMSProp.py`: Implements RMSProp optimization.
- `8-RMSProp.py`: TensorFlow-based RMSProp optimizer.
- `9-Adam.py`: Implements Adam optimization.
- `10-Adam.py`: TensorFlow-based Adam optimizer.
- `11-learning_rate_decay.py`: Implements inverse time decay in NumPy.
- `12-learning_rate_decay.py`: TensorFlow-based learning rate decay.
- `13-batch_norm.py`: Implements batch normalization for neural networks.
- `14-batch_norm.py`: TensorFlow-based batch normalization layer.
- `15-optimization_blog.md`: Blog post explaining optimization techniques.

## Usage
Each file can be run independently to test individual optimization techniques. The implementations can be integrated into deep learning workflows for improved training efficiency.

## License
This project is open-source and available for use and modification under the MIT License.
