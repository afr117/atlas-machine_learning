#!/usr/bin/env python3
"""
Convert a gensim Word2Vec model to a Keras Embedding layer
"""

import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a trained gensim Word2Vec model to a Keras Embedding layer.

    Args:
        model: a trained gensim Word2Vec model.

    Returns:
        tf.keras.layers.Embedding: a trainable Embedding layer whose weights
        are initialized from the gensim model's word vectors. The weights can
        be further updated during Keras training.
    """
    # gensim stores word vectors in wv.vectors with shape (vocab_size, vector_size)
    weights = model.wv.vectors
    vocab_size, vector_size = weights.shape

    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=[weights],
        trainable=True
    )

    return embedding_layer
