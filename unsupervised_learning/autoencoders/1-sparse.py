#!/usr/bin/env python3
"""Sparse autoencoder builder.

Adds L1 activity regularization on the latent representation to
encourage sparsity. Encoder/decoder are fully-connected and mirrored.
All hidden layers (incl. latent) use ReLU; decoder's final layer uses
Sigmoid. The full model is compiled with Adam and binary cross-entropy.
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Create a sparse (fully-connected) autoencoder.

    Args:
        input_dims (int): Dimensionality of the input.
        hidden_layers (list[int]): Units for each encoder hidden layer,
            in order (the decoder will mirror these in reverse).
        latent_dims (int): Size of the latent space (bottleneck).
        lambtha (float): L1 regularization strength applied to the
            latent layer's activations (activity regularizer).

    Returns:
        tuple:
            encoder (keras.Model): Inputs -> latent representation.
            decoder (keras.Model): Latent -> reconstructed input.
            auto (keras.Model): Full autoencoder, compiled.
    """
    # ----- Encoder -----
    enc_in = keras.Input(shape=(input_dims,), name="encoder_input")
    x = enc_in
    for i, units in enumerate(hidden_layers):
        x = keras.layers.Dense(
            units,
            activation="relu",
            name="enc_dense_{}".format(i)
        )(x)

    # Latent with L1 activity regularization (sparsity on activations)
    z = keras.layers.Dense(
        latent_dims,
        activation="relu",
        activity_regularizer=keras.regularizers.l1(lambtha),
        name="latent"
    )(x)
    encoder = keras.Model(enc_in, z, name="encoder")

    # ----- Decoder -----
    dec_in = keras.Input(shape=(latent_dims,), name="decoder_input")
    y = dec_in
    for i, units in enumerate(reversed(hidden_layers)):
        y = keras.layers.Dense(
            units,
            activation="relu",
            name="dec_dense_{}".format(i)
        )(y)

    dec_out = keras.layers.Dense(
        input_dims,
        activation="sigmoid",
        name="reconstruction"
    )(y)
    decoder = keras.Model(dec_in, dec_out, name="decoder")

    # ----- Autoencoder (encoder + decoder) -----
    auto_out = decoder(encoder(enc_in))
    auto = keras.Model(enc_in, auto_out, name="sparse_autoencoder")
    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
