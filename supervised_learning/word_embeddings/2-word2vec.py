#!/usr/bin/env python3
"""
Train a Word2Vec model using gensim
"""

import gensim


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
        sentences (list): list of sentences to train on.
            - Can be list of strings OR list of token lists.
        vector_size (int): dimensionality of embedding vectors.
        min_count (int): minimum word occurrences.
        window (int): context window size.
        negative (int): negative sampling size.
        cbow (bool): True = CBOW (sg=0); False = Skip-gram (sg=1).
        epochs (int): number of training iterations.
        seed (int): random seed.
        workers (int): number of worker threads.

    Returns:
        gensim.models.Word2Vec: trained Word2Vec model.
    """
    # If sentences are raw strings, split into tokens
    if len(sentences) > 0 and isinstance(sentences[0], str):
        sentences = [s.split() for s in sentences]

    sg = 0 if cbow else 1

    # Let gensim handle vocab building + training internally
    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers,
        epochs=epochs
    )

    return model
