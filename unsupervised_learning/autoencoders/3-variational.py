#!/usr/bin/env python3
"""Variational autoencoder (VAE) builder.

Encoder:
- Dense hidden layers with ReLU.
- Outputs: mean (mu) and log-variance (log_var) with linear activation.
- Samples latent z via the reparameterization trick.

Decoder:
- Mirror of encoder hidden sizes in reverse with ReLU.
- Final sigmoid layer to reconstruct input.

The model is compiled with Adam and binary cross-entropy. The KL term
is added to the model via a custom layer's add_loss, so compile() only
needs the reconstruction loss.
"""

import tensorflow.keras as keras


class _Sampler(keras.layers.Layer):
    """Reparameterization trick layer: z = mu + sigma * epsilon.

    Also adds the KL divergence term as a loss:
        KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    """

    def call(self, inputs):
        mu, log_var = inputs
        eps = keras.backend.random_normal(shape=keras.backend.shape(mu))
        z = mu + keras.backend.exp(0.5 * log_var) * eps

        kl = -0.5 * keras.backend.sum(
            1.0 + log_var - keras.backend.square(mu)
            - keras.backend.exp(log_var),
            axis=1
        )
        self.add_loss(keras.backend.mean(kl))
        return z


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Create a variational autoencoder (VAE).

    Args:
        input_dims (int): Dimensionality of the input.
        hidden_layers (list[int]): Units for each encoder hidden layer,
            in order (decoder mirrors these in reverse).
        latent_dims (int): Dimensionality of the latent space.

    Returns:
        tuple:
            encoder (keras.Model): Maps input -> (z, mu, log_var).
            decoder (keras.Model): Maps z -> reconstruction.
            auto (keras.Model): Full VAE (compiled).
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

    z = _Sampler(name="sampler")([mu, log_var])

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
    auto_out = decoder(encoder(enc_in)[0])
    auto = keras.Model(enc_in, auto_out, name="variational_autoencoder")
    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
