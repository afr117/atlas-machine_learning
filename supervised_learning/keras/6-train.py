#!/usr/bin/env python3
"""
Trains a Keras model using mini-batch gradient descent
with validation and early stopping.
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent with
    validation and optional early stopping.

    Args:
        network (keras.Model): The model to train.
        data (numpy.ndarray): Input data of shape (m, nx).
        labels (numpy.ndarray): One-hot labels of shape (m, classes).
        batch_size (int): Batch size for mini-batch gradient descent.
        epochs (int): Number of training epochs.
        validation_data (tuple, optional): Data to validate the model with.
        Defaults to None.
        early_stopping (bool, optional): Whether to apply early stopping.
        Defaults to False.
        patience (int, optional): Number of epochs to wait before
        stopping if no improvement. Defaults to 0.
        verbose (bool, optional): Whether to print training output.
        Defaults to True.
        shuffle (bool, optional): Whether to shuffle data every epoch.
        Defaults to False.

    Returns:
        keras.callbacks.History: The history object generated after training.
    """
    callbacks = []
    if early_stopping and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience)
        callbacks.append(early_stop)

    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle,
        callbacks=callbacks
    )
