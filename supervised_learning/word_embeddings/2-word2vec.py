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
        sentences (list): list of sentences to train on
            - If a list of strings is provided, they will be split into words.
        vector_size (int): dimensionality of the embedding vectors
        min_count (int): minimum number of occurrences to include a word
        window (int): maximum distance between predicted word and context
        negative (int): number of negative samples for training
        cbow (bool): True = CBOW; False = Skip-gram
        epochs (int): number of training iterations
        seed (int): random seed for reproducibility
        workers (int): number of worker threads

    Returns:
        gensim.models.Word2Vec: trained Word2Vec model
    """
    # If input sentences are raw strings, tokenize
    if len(sentences) > 0 and isinstance(sentences[0], str):
        sentences = [s.split() for s in sentences]

    sg = 0 if cbow else 1

    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers
    )

    model.train(
        corpus_iterable=sentences,
        total_examples=len(sentences),
        epochs=epochs
    )

    return model
