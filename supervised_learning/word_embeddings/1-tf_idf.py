#!/usr/bin/env python3
"""
TF-IDF embedding
"""

import numpy as np


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix.

    Args:
        sentences (list of str): sentences to analyze.
        vocab (list of str): optional vocabulary list.
            If None, the vocabulary is built from all sentences.

    Returns:
        embeddings (np.ndarray): shape (s, f), TF-IDF embeddings.
            s is the number of sentences.
            f is the number of features.
        features (np.ndarray): shape (f,), list of feature words
            corresponding to the columns of embeddings.
    """
    def preprocess(token):
        """
        Normalize a token: lowercase, strip punctuation, handle possessives.
        """
        token = token.lower()
        # remove common punctuation at the start and end
        token = token.strip(".,!?;:-()[]{}\"")
        # handle possessives like "children's" -> "children"
        if token.endswith("'s"):
            token = token[:-2]
        # remove trailing single quote if any
        if token.endswith("'"):
            token = token[:-1]
        return token

    # ----- Build vocabulary / features -----
    if vocab is None:
        vocab_set = set()
        for sentence in sentences:
            for tok in sentence.split():
                word = preprocess(tok)
                if word:
                    vocab_set.add(word)
        features_list = sorted(vocab_set)
    else:
        # keep the provided order
        features_list = list(vocab)

    features = np.array(features_list)

    # Map words to indices
    word_index = {w: i for i, w in enumerate(features)}

    s = len(sentences)
    f = len(features)

    # ----- Term Frequency (TF): raw counts -----
    tf = np.zeros((s, f), dtype=float)
    for i, sentence in enumerate(sentences):
        for tok in sentence.split():
            word = preprocess(tok)
            if word in word_index:
                j = word_index[word]
                tf[i, j] += 1.0

    # ----- Document Frequency (DF) -----
    # number of sentences where the term appears
    df = np.count_nonzero(tf > 0, axis=0)

    # ----- Inverse Document Frequency (IDF) -----
    # sklearn-like: idf = log((1 + N) / (1 + df)) + 1
    N = float(s)
    idf = np.log((1.0 + N) / (1.0 + df)) + 1.0

    # terms that never appear will have df=0 but tf=0, so column stays 0
    tf_idf_matrix = tf * idf

    # ----- L2 normalization per sentence -----
    norms = np.linalg.norm(tf_idf_matrix, axis=1, keepdims=True)
    # avoid division by zero: only normalize non-zero rows
    nonzero = norms != 0
    tf_idf_matrix[nonzero] = tf_idf_matrix[nonzero] / norms[nonzero]

    return tf_idf_matrix, features
