#!/usr/bin/env python3
"""
Bag of Words embedding
"""

import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Args:
        sentences (list of str): sentences to analyze.
        vocab (list of str): optional vocabulary to use. If None, the
            vocabulary is built from all sentences.

    Returns:
        embeddings (np.ndarray): shape (s, f) with word-count embeddings
            for each sentence.
        features (np.ndarray): shape (f,) array of the feature words in
            the same order as the columns of embeddings.
    """
    def preprocess(token):
        """
        Normalize a token: lowercase, strip punctuation, handle 's.
        """
        token = token.lower()
        # remove common punctuation at the start/end
        token = token.strip(".,!?;:-()[]{}\"")
        # handle possessives like "children's" -> "children"
        if token.endswith("'s"):
            token = token[:-2]
        # remove trailing single quote if any remains
        if token.endswith("'"):
            token = token[:-1]
        return token

    # Build vocabulary if none is provided
    if vocab is None:
        vocab_set = set()
        for sentence in sentences:
            for tok in sentence.split():
                word = preprocess(tok)
                if word:
                    vocab_set.add(word)
        # sort to have a consistent, deterministic order
        features_list = sorted(vocab_set)
    else:
        # keep the given order if vocab is provided
        features_list = list(vocab)

    # Turn features into a numpy array (matches example printout)
    features = np.array(features_list)

    # Map each word to its column index
    word_index = {word: idx for idx, word in enumerate(features)}

    # Initialize embeddings matrix
    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    # Fill the matrix with word counts
    for i, sentence in enumerate(sentences):
        for tok in sentence.split():
            word = preprocess(tok)
            if word in word_index:
                j = word_index[word]
                embeddings[i, j] += 1

    return embeddings, features
