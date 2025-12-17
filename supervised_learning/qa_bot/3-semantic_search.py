#!/usr/bin/env python3
"""
Module for Semantic Search using TensorFlow Hub
"""
import numpy as np
import os
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents
    """
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    articles = []
    filenames = sorted(os.listdir(corpus_path))

    for filename in filenames:
        if filename.endswith('.md'):
            path = os.path.join(corpus_path, filename)
            with open(path, 'r', encoding='utf-8') as f:
                articles.append(f.read())

    # We embed the query and all documents
    embeddings = model([sentence] + articles)
    
    # query_vec is index 0, doc_vecs are index 1 onwards
    query_vec = embeddings[0:1]
    doc_vecs = embeddings[1:]

    # Calculate dot product
    # We use np.inner and then flatten to get a 1D array of scores
    similarities = np.inner(query_vec, doc_vecs).flatten()

    # Find the index of the best match
    best_match_idx = np.argmax(similarities)

    return articles[best_match_idx]
