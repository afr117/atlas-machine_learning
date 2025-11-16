#!/usr/bin/env python3
"""
Train a Word2Vec model using gensim
"""

from gensim.models import Word2Vec


def word2vec_model(
    sentences,
    vector_size=100,
    min_count=5,
    window=5,
    negative=5,
    cbow=True,
    epochs=5,
    seed=0,
    workers=1
):
    """
    Creates, builds, and trains a gensim Word2Vec model.

    Args:
        sentences (list): Sentences to be trained on.
            - Expected: list of list of tokens (e.g. [["hello", "world"], ...])
            - If a list of strings is provided, they will be split on whitespace.
        vector_size (int): Dimensionality of the embedding vectors.
        min_count (int): Minimum number of occurrences of a word to be kept.
        window (int): Maximum distance between the current and predicted word.
        negative (int): Number of negative samples.
        cbow (bool): If True, train using CBOW (sg=0); if False, use Skip-gram (sg=1).
        epochs (int): Number of training iterations.
        seed (int): Random seed for reproducibility.
        workers (int): Number of worker threads to train the model.

    Returns:
        gensim.models.Word2Vec: The trained Word2Vec model.
    """
    # If sentences are given as list of raw strings, tokenize by simple split
    if len(sentences) > 0 and isinstance(sentences[0], str):
        sentences = [s.split() for s in sentences]

    sg = 0 if cbow else 1

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        workers=workers,
        seed=seed,
        epochs=epochs
    )

    return model
