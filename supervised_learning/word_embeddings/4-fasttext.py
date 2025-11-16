#!/usr/bin/env python3
"""
Train a FastText model using gensim
"""

import gensim


def fasttext_model(
    sentences,
    vector_size=100,
    min_count=5,
    negative=5,
    window=5,
    cbow=True,
    epochs=5,
    seed=0,
    workers=1
):
    """
    Creates, builds, and trains a gensim FastText model.

    Args:
        sentences (list): list of sentences to train on.
            Can be a list of strings or a list of token lists.
        vector_size (int): dimensionality of the embedding vectors.
        min_count (int): minimum number of occurrences of a word
            to be included in the vocabulary.
        negative (int): number of negative samples.
        window (int): maximum distance between the current and
            predicted word within a sentence.
        cbow (bool): True = CBOW (sg=0); False = Skip-gram (sg=1).
        epochs (int): number of training iterations.
        seed (int): random seed for reproducibility.
        workers (int): number of worker threads used during training.

    Returns:
        gensim.models.FastText: trained FastText model.
    """
    # If input is a list of raw strings, split into tokens
    if len(sentences) > 0 and isinstance(sentences[0], str):
        sentences = [s.split() for s in sentences]

    # sg = 0 for CBOW, 1 for Skip-gram
    sg = 0 if cbow else 1

    # Let gensim handle vocabulary building + training internally
    model = gensim.models.FastText(
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
