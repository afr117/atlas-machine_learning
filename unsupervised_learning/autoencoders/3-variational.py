#!/usr/bin/env python3
"""Variational autoencoder (VAE) with graph-attached KL loss.

- Encoder: hidden Dense layers (ReLU), then mean (mu) and log-variance
  (log_var) with linear activation (None). Latent z sampled via
  reparameterization.
- Decoder: hidden layers mirrored (ReLU), final Dense(sigmoid).
- KL divergence is added through a custom Layer that lives in the graph,
  so `auto.losses` is populated immediately after model creation.
"""

import tensorflow.keras as keras


class KLLoss(keras.layers.Layer):
    """Adds KL divergence to the model via `add_loss`.

    KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var)) per sample
    """

    def call(self, inputs):
        mu, log_var = inputs
        kl = -0.5 * keras.backend.sum(
            1.0 + log_var
            - keras.backend.square(mu)
            - keras.backend.exp(log_var),
            axis=1
        )
        self.add_loss(keras.backend.mean(kl))
        # Pass-through (unused); returning mu keeps shape sane if needed
        return mu


def _reparameterize(args):
    """z = mu + exp(0.5*log_var) * eps."""
    mu, log_var = args
    eps = keras.backend.random_normal(shape=keras.backend.shape(mu))
    return mu + keras.backend.exp(0.5 * log_var) * eps


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Create a variational autoencoder (VAE).

    Args:
        input_dims (int): Input dimensionality.
        hidden_layers (list[int]): Encoder hidden units (decoder mirrors).
        latent_dims (int): Latent dimensionality.

    Returns:
        tuple:
            encoder (keras.Model): input -> (z, mu, log_var).
            decoder (keras.Model): z -> reconstruction.
            auto (keras.Model): full VAE, compiled (Adam + BCE).
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

    # Attach KL via a graph layer so model.losses is populated
    _ = KLLoss(name="kl_loss")([mu, log_var])

    z = keras.layers.Lambda(_reparameterize, name="z")([mu, log_var])

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

    # ----- Full VAE -----
    z_out, mu_out, log_var_out = encoder(enc_in)
    recon = decoder(z_out)
    auto = keras.Model(enc_in, recon, name="variational_autoencoder")

    # Reconstruction loss only; KL is already in `auto.losses` via KLLoss
    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
