#!/usr/bin/env python3
"""
Trains a Keras model using mini-batch gradient descent with validation,
early stopping, learning rate decay, and saving the best model.
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, validation_data=None,
                early_stopping=False, patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent with
    validation, early stopping,
    learning rate decay, and model checkpointing.

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
        patience (int, optional): Number of epochs to wait before stopping if no improvement. Defaults to 0.
        learning_rate_decay (bool, optional): Whether to use learning rate decay.
        Defaults to False.
        alpha (float, optional): Initial learning rate. Defaults to 0.1.
        decay_rate (float, optional): Decay rate for learning rate.
        Defaults to 1.
        save_best (bool, optional): Whether to save the best model
        based on validation loss. Defaults to False.
        filepath (str, optional): File path to save the best model.
        Defaults to None.
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

    if learning_rate_decay and validation_data is not None:
        def lr_schedule(epoch):
            return alpha / (1 + decay_rate * epoch)
        lr_decay = K.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
        callbacks.append(lr_decay)

    if save_best and validation_data is not None and filepath:
        model_checkpoint = K.callbacks.ModelCheckpoint(filepath=filepath,
                                                       save_best_only=True,
                                                       monitor='val_loss')
        callbacks.append(model_checkpoint)

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
