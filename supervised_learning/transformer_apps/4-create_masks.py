#!/usr/bin/env python3
"""
Module defining the create_masks function.
"""
import tensorflow as tf


def create_padding_mask(seq):
    """
    Creates a padding mask for a batch of sequences.

    Args:
        seq: A tf.Tensor of shape (batch_size, seq_len).

    Returns:
        mask: A tf.Tensor padding mask of shape (batch_size, 1, 1, seq_len).
    """
    # Find all sequence elements that are padding (value 0).
    mask = tf.cast(tf.equal(seq, 0), tf.float32)

    # Return shape (batch_size, 1, 1, seq_len) for broadcasting in attention.
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(seq_len):
    """
    Creates a look-ahead mask for masking future tokens in a sequence.

    Args:
        seq_len: The length of the sequence (integer).

    Returns:
        mask: A tf.Tensor look-ahead mask of shape (seq_len, seq_len).
    """
    # Create a lower triangular matrix of ones.
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return mask  # This mask is (seq_len, seq_len)


def create_masks(inputs, target):
    """
    Creates all masks for training/validation.

    Args:
        inputs: tf.Tensor of shape (batch_size, seq_len_in)
        target: tf.Tensor of shape (batch_size, seq_len_out)

    Returns:
        encoder_mask: tf.Tensor padding mask (batch_size, 1, 1, seq_len_in)
        combined_mask: tf.Tensor combined mask (batch_size, 1, seq_len_out, seq_len_out)
        decoder_mask: tf.Tensor padding mask (batch_size, 1, 1, seq_len_in)
    """
    # --- 1. Encoder Padding Mask (For Encoder Self-Attention) ---
    # Masks padding in the inputs sequence.
    encoder_mask = create_padding_mask(inputs)

    # --- 2. Decoder Padding Mask (For Encoder-Decoder Attention) ---
    # Masks padding in the inputs sequence (same as encoder mask).
    decoder_mask = create_padding_mask(inputs)

    # --- 3. Combined Mask (For Decoder Self-Attention) ---
    
    # a. Decoder Target Padding Mask
    # Masks padding in the target sequence. Shape: (batch_size, 1, 1, seq_len_out)
    target_padding_mask = create_padding_mask(target)
    
    # b. Look-ahead Mask
    # Masks future tokens. Shape: (seq_len_out, seq_len_out)
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
    
    # c. Combine the masks
    # The combined mask is the maximum of the look-ahead mask and the target padding mask.
    # The target padding mask (1, 1, 1, seq_len_out) is broadcasted to (1, 1, seq_len_out, seq_len_out).
    # tf.maximum(look_ahead_mask, target_padding_mask)
    combined_mask = tf.maximum(target_padding_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask
