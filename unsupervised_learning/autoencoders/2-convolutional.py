#!/usr/bin/env python3
"""Convolutional autoencoder builder.

Encoder:
- Repeated blocks: Conv2D(filters[i], 3x3, padding='same', relu)
  then MaxPooling2D(2x2, padding='same').

Decoder:
- For all but the last two convs: Conv2D(3x3, padding='same', relu)
  then UpSampling2D(2x2).
- Second-to-last conv: Conv2D(3x3, padding='valid', relu) (followed by
  an upsampling to align spatial dims).
- Last conv: Conv2D(channels, 3x3, padding='same', sigmoid), no upsample.

Full model compiled with Adam and binary cross-entropy.
"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Create a convolutional autoencoder.

    Args:
        input_dims (tuple[int]): Input H, W, C.
        filters (list[int]): Conv filters for encoder (per block).
        latent_dims (tuple[int]): Latent H, W, C representation.

    Returns:
        tuple:
            encoder (keras.Model): Inputs -> latent feature map.
            decoder (keras.Model): Latent -> reconstructed image.
            auto (keras.Model): Full autoencoder, compiled.
    """
    # ----- Encoder -----
    enc_in = keras.Input(shape=input_dims, name="encoder_input")
    x = enc_in
    for i, f in enumerate(filters):
        x = keras.layers.Conv2D(
            f, (3, 3), activation="relu", padding="same",
            name="enc_conv_{}".format(i)
        )(x)
        x = keras.layers.MaxPooling2D(
            (2, 2), padding="same", name="enc_pool_{}".format(i)
        )(x)

    # Latent output
    encoder = keras.Model(enc_in, x, name="encoder")

    # ----- Decoder -----
    dec_in = keras.Input(shape=latent_dims, name="decoder_input")
    y = dec_in
    rev = list(reversed(filters))
    n = len(rev)

    for i, f in enumerate(rev):
        # Index within decoder conv stack (excluding final output conv)
        is_last_two = (i >= n - 2)
        if not is_last_two:
            # Conv (same) + UpSampling
            y = keras.layers.Conv2D(
                f, (3, 3), activation="relu", padding="same",
                name="dec_conv_{}".format(i)
            )(y)
            y = keras.layers.UpSampling2D(
                (2, 2), name="dec_ups_{}".format(i)
            )(y)
        elif i == n - 2:
            # Second-to-last conv: valid padding to adjust dims,
            # then upsample to align with target spatial size.
            y = keras.layers.Conv2D(
                f, (3, 3), activation="relu", padding="valid",
                name="dec_conv_{}".format(i)
            )(y)
            y = keras.layers.UpSampling2D(
                (2, 2), name="dec_ups_{}".format(i)
            )(y)
        else:
            # 'i == n - 1' handled after loop by final output conv
            pass

    # Final output conv: match input channels, sigmoid, no upsample
    out = keras.layers.Conv2D(
        input_dims[-1], (3, 3), activation="sigmoid", padding="same",
        name="decoder_output"
    )(y)

    decoder = keras.Model(dec_in, out, name="decoder")

    # ----- Autoencoder (encoder + decoder) -----
    auto_out = decoder(encoder(enc_in))
    auto = keras.Model(enc_in, auto_out, name="conv_autoencoder")
    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
