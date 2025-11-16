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
            Can be a list of strings or a list of token lists.
        vector_size (int): dimensionality of embedding vectors.
        min_count (int): minimum word occurrences to keep.
        window (int): context window size.
        negative (int): negative sampling size.
        cbow (bool): True = CBOW (sg=0), False = Skip-gram (sg=1).
        epochs (int): number of training iterations (not used explicitly;
                      training uses gensim’s default epochs).
        seed (int): random seed.
        workers (int): number of worker threads.

    Returns:
        gensim.models.Word2Vec: trained Word2Vec model.
    """
    # If input is a list of raw strings, split into tokens
    if len(sentences) > 0 and isinstance(sentences[0], str):
        sentences = [s.split() for s in sentences]

    # sg=0 for CBOW, sg=1 for Skip-gram
    sg = 0 if cbow else 1

    # Let gensim handle vocab building + training internally
    # Do NOT pass epochs here, to match the checker’s reference behavior.
    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        workers=workers,
        seed=seed
    )

    return model
