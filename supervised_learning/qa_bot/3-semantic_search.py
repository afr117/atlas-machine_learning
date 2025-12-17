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
    # Load the Universal Sentence Encoder from TF Hub
    model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(model_url)

    # Prepare list to hold document texts and titles
    articles = []

    # Iterate through the files in the corpus directory
    for filename in os.listdir(corpus_path):
        if not filename.endswith('.md'):
            continue
        
        path = os.path.join(corpus_path, filename)
        with open(path, 'r', encoding='utf-8') as f:
            articles.append(f.read())

    # Add the query sentence to the list to embed everything at once
    # Or embed separately:
    documents_embeddings = model(articles)
    query_embedding = model([sentence])

    # Calculate Cosine Similarity
    # Similarity = (A dot B) / (||A|| * ||B||)
    # Since USE outputs normalized vectors, we can just use dot product
    similarities = np.inner(query_embedding, documents_embeddings)

    # Find the index of the highest similarity score
    closest_idx = np.argmax(similarities)

    return articles[closest_idx]
