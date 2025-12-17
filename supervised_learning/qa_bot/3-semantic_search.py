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

    Args:
        corpus_path (str): Path to the folder of reference documents
        sentence (str): The query sentence to search for

    Returns:
        str: The text of the document most similar to the sentence
    """
    # Load the Universal Sentence Encoder (USE)
    model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(model_url)

    articles = []
    
    # Sort the filenames to ensure consistent indexing
    filenames = sorted(os.listdir(corpus_path))

    for filename in filenames:
        if filename.endswith('.md'):
            path = os.path.join(corpus_path, filename)
            # Use utf-8 to handle any special characters in markdown
            with open(path, 'r', encoding='utf-8') as f:
                articles.append(f.read())

    # Generate embeddings for the documents and the query
    # model() expects a list of strings
    doc_embeddings = model(articles)
    query_embedding = model([sentence])

    # Calculate cosine similarity using inner product (dot product)
    # USE vectors are already normalized (length = 1)
    similarities = np.inner(query_embedding, doc_embeddings)

    # argmax returns the index of the highest score
    # similarities is shape (1, num_articles), so we take [0]
    closest_idx = np.argmax(similarities[0])

    return articles[closest_idx]
