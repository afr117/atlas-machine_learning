#!/usr/bin/env python3
"""Bayesian Optimization of Keras Model using GPyOpt"""

import numpy as np
import GPy
import GPyOpt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split and scale data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Training function
def build_and_train_model(params):
    params = params[0]
    learning_rate = float(params[0])
    units = int(params[1])
    dropout = float(params[2])
    l2 = float(params[3])
    batch_size = int(params[4])

    tf.keras.backend.clear_session()
    model = Sequential([
        Dense(units, activation='relu', input_shape=(X_train.shape[1],),
              kernel_regularizer=tf.keras.regularizers.l2(l2)),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])

    filename = f'checkpoint_lr{learning_rate:.5f}_u{units}_d{dropout:.2f}_l2{l2:.5f}_b{batch_size}.h5'
    checkpoint = ModelCheckpoint(filename, monitor='val_accuracy', save_best_only=True,
                                 save_weights_only=False, verbose=0, mode='max')
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, mode='max',
                               restore_best_weights=True, verbose=0)

    model.fit(X_train, y_train, epochs=50, batch_size=batch_size,
              validation_data=(X_val, y_val), callbacks=[checkpoint, early_stop],
              verbose=0)

    preds = model.predict(X_val)
    preds = (preds > 0.5).astype(int)
    acc = accuracy_score(y_val, preds)
    return -acc  # GPyOpt minimizes

# Hyperparameter bounds
bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-4, 1e-2)},
    {'name': 'units', 'type': 'discrete', 'domain': (16, 32, 64, 128)},
    {'name': 'dropout', 'type': 'continuous', 'domain': (0.0, 0.5)},
    {'name': 'l2', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64)}
]

# Run Bayesian Optimization
opt = GPyOpt.methods.BayesianOptimization(
    f=build_and_train_model,
    domain=bounds,
    acquisition_type='EI',
    exact_feval=True,
    maximize=False
)

opt.run_optimization(max_iter=30)

# Save convergence plot
opt.plot_convergence()
plt.savefig("convergence_plot.png")

# Save summary report
with open("bayes_opt.txt", "w") as f:
    f.write("Best parameters found:\n")
    f.write(str(opt.X[opt.index_minimum]) + "\n")
    f.write(f"Best accuracy: {-opt.fx_opt:.4f}\n")
