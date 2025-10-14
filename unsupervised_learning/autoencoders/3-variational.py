#!/usr/bin/env python3
"""Variational autoencoder (VAE).

Encoder:
- Dense hidden layers (ReLU), then mean (mu) and log-variance (log_var)
  with linear activation (None).
- Reparameterization: z = mu + exp(0.5 * log_var) * epsilon.

Decoder:
- Hidden layers mirrored (ReLU), final layer Sigmoid to reconstruct.

The full model is compiled with Adam + binary cross-entropy
(reconstruction). The KL divergence is added to the full model via
`auto.add_loss(...)` so tests see it in `auto.losses`.
"""

import tensorflow.keras as keras


def _sample_z(args):
    """Reparameterization trick: sample z ~ N(mu, sigma^2)."""
    mu, log_var = args
    eps = keras.backend.random_normal(shape=keras.backend.shape(mu))
    return mu + keras.backend.exp(0.5 * log_var) * eps


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Create a variational autoencoder (VAE).

    Args:
        input_dims (int): Dimensionality of the input.
        hidden_layers (list[int]): Units for encoder hidden layers
            (decoder mirrors in reverse).
        latent_dims (int): Size of the latent space.

    Returns:
        tuple:
            encoder (keras.Model): input -> (z, mu, log_var).
            decoder (keras.Model): z -> reconstruction.
            auto (keras.Model): full VAE (compiled).
    """
    # ----- Encoder -----
    enc_in = keras.Input(shape=(input_dims,), name="encoder_input")
    x = enc_in
    for i, units in enumerate(hidden_layers):
        x = keras.layers.Dense(
            units, activation="relu", name="enc_dense_{}".format(i)
        )(x)

    mu = keras.layers.Dense(latent_dims, activation=None, name="mu")(x)
    log_var = keras.layers.Dense(
        latent_dims, activation=None, name="log_var"
    )(x)

    z = keras.layers.Lambda(_sample_z, name="z")([mu, log_var])

    encoder = keras.Model(enc_in, [z, mu, log_var], name="encoder")

    # ----- Decoder -----
    dec_in = keras.Input(shape=(latent_dims,), name="decoder_input")
    y = dec_in
    for i, units in enumerate(reversed(hidden_layers)):
        y = keras.layers.Dense(
            units, activation="relu", name="dec_dense_{}".format(i)
        )(y)

    dec_out = keras.layers.Dense(
        input_dims, activation="sigmoid", name="reconstruction"
    )(y)

    decoder = keras.Model(dec_in, dec_out, name="decoder")

    # ----- Full VAE (attach KL at model level) -----
    z_out, mu_out, log_var_out = encoder(enc_in)
    recon = decoder(z_out)
    auto = keras.Model(enc_in, recon, name="variational_autoencoder")

    # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl = -0.5 * keras.backend.sum(
        1.0 + log_var_out
        - keras.backend.square(mu_out)
        - keras.backend.exp(log_var_out),
        axis=1
    )
    auto.add_loss(keras.backend.mean(kl))

    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
