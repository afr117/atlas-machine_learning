#!/usr/bin/env python3
"""
Function to perform semantic search on a corpus of documents using Universal Sentence Encoder.
"""
import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents.

    Args:
        corpus_path (str): The path to the corpus of reference documents 
                           on which to perform semantic search.
        sentence (str): The sentence from which to perform semantic search.

    Returns:
        str: The reference text of the document most similar to sentence,
             or None if the model or documents cannot be loaded.
    """
    # Load the Universal Sentence Encoder (USE) model from TensorFlow Hub.
    try:
        # We use a non-portable model handle for robustness, though a larger 
        # model (like 'universal-sentence-encoder-large/5') could also be used.
        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # 1. Read Corpus Documents
    documents = []
    
    # Iterate through all files in the corpus directory
    for filename in os.listdir(corpus_path):
        filepath = os.path.join(corpus_path, filename)
        
        if os.path.isfile(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(content)
            except Exception as e:
                # Log error but continue with other files
                print(f"Could not read file {filename}: {e}")

    if not documents:
        print("No documents found in the corpus path.")
        return None

    # 2. Generate Embeddings
    # The list of texts to embed: [query, doc1, doc2, ...]
    all_texts = [sentence] + documents

    try:
        # Embed the query and all documents in one batch for efficiency
        embeddings = embed(all_texts)
    except Exception as e:
        print(f"Error during embedding generation: {e}")
        return None
    
    # Separate the query embedding from the document embeddings
    query_embedding = embeddings[0:1] # Shape (1, 512)
    document_embeddings = embeddings[1:] # Shape (N, 512)

    # 3. Calculate Cosine Similarity
    # Compare the query vector to every document vector
    similarity_matrix = cosine_similarity(query_embedding, document_embeddings)

    # 4. Find the Best Match
    # np.argmax finds the index of the highest similarity score
    best_match_index = np.argmax(similarity_matrix[0])

    # 5. Return the Content of the Best Document
    return documents[best_match_index]
