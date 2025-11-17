#!/usr/bin/env python3
"""
Module defining the Transformer model components.
Based on the "Attention is All You Need" paper.
"""
import tensorflow as tf
import numpy as np


def sdp_attention(q, k, v, mask):
    """
    Calculates the Scaled Dot Product Attention.
    
    Args:
        q: query tensor, shape (..., seq_len_q, depth)
        k: key tensor, shape (..., seq_len_k, depth)
        v: value tensor, shape (..., seq_len_v, depth_v)
        mask: mask tensor, shape (..., seq_len_q, seq_len_k)
    
    Returns:
        output: tensor, shape (..., seq_len_q, depth_v)
        weights: tensor, shape (..., seq_len_q, seq_len_k)
    """
    # Matmul Q and K transpose
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # Scale by the square root of the depth
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Add the mask to the scaled tensor.
    if mask is not None:
        # Add a large negative number (e.g., -1e9) to masked elements 
        # so they become 0 when softmax is applied.
        scaled_attention_logits += (mask * -1e9)  

    # Softmax on the last axis (seq_len_k) to get attention weights
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Matmul the attention weights and V
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    MultiHeadAttention layer.
    """
    def __init__(self, dm, h):
        """
        Initializes the Multi-Head Attention layer.
        
        Args:
            dm (int): Dimensionality of the model.
            h (int): Number of heads.
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm

        # Ensure dm is divisible by h
        self.depth = dm // h

        self.wq = tf.keras.layers.Dense(dm)
        self.wk = tf.keras.layers.Dense(dm)
        self.wv = tf.keras.layers.Dense(dm)

        self.dense = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # 1. Linear layers for Q, K, V
        q = self.wq(q)  # (batch_size, seq_len, dm)
        k = self.wk(k)  # (batch_size, seq_len, dm)
        v = self.wv(v)  # (batch_size, seq_len, dm)

        # 2. Split heads
        q = self.split_heads(q, batch_size)  # (batch_size, h, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, h, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, h, seq_len_v, depth)

        # 3. Scaled Dot Product Attention
        scaled_attention, attention_weights = sdp_attention(q, k, v, mask)
        # scaled_attention shape: (batch_size, h, seq_len_q, depth)

        # 4. Concatenate heads
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # scaled_attention shape: (batch_size, seq_len_q, h, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.dm))  # (batch_size, seq_len_q, dm)

        # 5. Final linear layer
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, dm)

        return output, attention_weights


def point_wise_feed_forward_network(dm, hidden):
    """
    Creates the Point Wise Feed Forward Network (FFN).
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(hidden, activation='relu'),  # (batch_size, seq_len, hidden)
        tf.keras.layers.Dense(dm)  # (batch_size, seq_len, dm)
    ])


class EncoderBlock(tf.keras.layers.Layer):
    """
    One block of the Transformer Encoder.
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.ffn = point_wise_feed_forward_network(dm, hidden)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        # Multi-Head Attention (Self-Attention)
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, dm)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, dm)

        # Feed Forward Network
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, dm)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, dm)

        return out2


class DecoderBlock(tf.keras.layers.Layer):
    """
    One block of the Transformer Decoder.
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)  # Decoder self-attention
        self.mha2 = MultiHeadAttention(dm, h)  # Encoder-Decoder attention

        self.ffn = point_wise_feed_forward_network(dm, hidden)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # 1. Masked Multi-Head Attention (Self-Attention)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        # 2. Multi-Head Attention (Encoder-Decoder Attention)
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask
        )  # (batch_size, target_seq_len, dm)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)  # (batch_size, target_seq_len, dm)

        # 3. Feed Forward Network
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, dm)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)  # (batch_size, target_seq_len, dm)

        return out3, attn_weights_block1, attn_weights_block2


def get_angles(pos, i, dm):
    """Calculates the angles for the positional encoding."""
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / dm)
    return pos * angle_rates


def positional_encoding(position, dm):
    """Creates the positional encoding matrix."""
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(dm)[np.newaxis, :],
                            dm)

    # Apply sin to even indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Apply cos to odd indices
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class Encoder(tf.keras.layers.Layer):
    """
    The full Transformer Encoder.
    """
    def __init__(self, N, dm, h, hidden, input_vocab_size, max_seq_len, drop_rate=0.1):
        super(Encoder, self).__init__()
        self.dm = dm
        self.N = N
        
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, dm)
        self.pos_encoding = positional_encoding(max_seq_len, dm)
        
        self.enc_blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                           for _ in range(N)]
        
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # 1. Embedding + Positional Encoding
        x = self.embedding(x)  # (batch_size, input_seq_len, dm)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        # 2. Dropout
        x = self.dropout(x, training=training)

        # 3. Encoder Blocks
        for i in range(self.N):
            x = self.enc_blocks[i](x, training, mask)

        return x  # (batch_size, input_seq_len, dm)


class Decoder(tf.keras.layers.Layer):
    """
    The full Transformer Decoder.
    """
    def __init__(self, N, dm, h, hidden, target_vocab_size, max_seq_len, drop_rate=0.1):
        super(Decoder, self).__init__()
        self.dm = dm
        self.N = N
        
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, dm)
        self.pos_encoding = positional_encoding(max_seq_len, dm)
        
        self.dec_blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                           for _ in range(N)]
        
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        # 1. Embedding + Positional Encoding
        x = self.embedding(x)  # (batch_size, target_seq_len, dm)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        # 2. Dropout
        x = self.dropout(x, training=training)

        # 3. Decoder Blocks
        for i in range(self.N):
            x, block1, block2 = self.dec_blocks[i](
                x, enc_output, training, look_ahead_mask, padding_mask
            )

            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        # x shape: (batch_size, target_seq_len, dm)
        return x, attention_weights


class Transformer(tf.keras.Model):
    """
    The main Transformer model.
    """
    def __init__(self, N, dm, h, hidden, input_vocab_size, 
                 target_vocab_size, max_seq_len, drop_rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(N, dm, h, hidden, input_vocab_size,
                               max_seq_len, drop_rate)

        self.decoder = Decoder(N, dm, h, hidden, target_vocab_size,
                               max_seq_len, drop_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, input_sequence, target_sequence, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        
        # 1. Encoder output
        enc_output = self.encoder(input_sequence, training, encoder_mask)  # (batch_size, inp_seq_len, dm)

        # 2. Decoder output
        dec_output, attention_weights = self.decoder(
            target_sequence, enc_output, training, look_ahead_mask, decoder_mask
        )  # (batch_size, tar_seq_len, dm)

        # 3. Final linear layer
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights
