#!/usr/bin/env python3
"""Vanilla autoencoder builder.

Creates an autoencoder made of an encoder and a mirrored decoder.
All hidden layers (including the latent layer) use ReLU; the decoder's
final layer uses Sigmoid. The full model is compiled with Adam and
binary cross-entropy.

Allowed import: `import tensorflow.keras as keras`.
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Create a vanilla (fully-connected) autoencoder.

    Args:
        input_dims (int): Dimensionality of the input.
        hidden_layers (list of int): Units for each encoder hidden layer,
            in order (the decoder will mirror these in reverse).
        latent_dims (int): Size of the latent space (bottleneck).

    Returns:
        (encoder, decoder, auto):
            encoder (keras.Model): Maps inputs -> latent representation.
            decoder (keras.Model): Maps latent -> reconstructed input.
            auto (keras.Model): Full autoencoder (input -> reconstruction),
                compiled with Adam and binary cross-entropy.
    """
    # ----- Encoder -----
    enc_in = keras.Input(shape=(input_dims,), name="encoder_input")
    x = enc_in
    for i, units in enumerate(hidden_layers):
        x = keras.layers.Dense(
            units, activation="relu", name=f"enc_dense_{i}"
        )(x)

    # Latent (bottleneck)
    z = keras.layers.Dense(
        latent_dims, activation="relu", name="latent"
    )(x)

    encoder = keras.Model(enc_in, z, name="encoder")

    # ----- Decoder -----
    dec_in = keras.Input(shape=(latent_dims,), name="decoder_input")
    y = dec_in
    for i, units in enumerate(reversed(hidden_layers)):
        y = keras.layers.Dense(
            units, activation="relu", name=f"dec_dense_{i}"
        )(y)

    dec_out = keras.layers.Dense(
        input_dims, activation="sigmoid", name="reconstruction"
    )(y)

    decoder = keras.Model(dec_in, dec_out, name="decoder")

    # ----- Autoencoder (encoder + decoder) -----
    auto_out = decoder(encoder(enc_in))
    auto = keras.Model(enc_in, auto_out, name="autoencoder")
    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto

