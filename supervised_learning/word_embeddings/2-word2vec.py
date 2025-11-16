#!/usr/bin/env python3
import gensim

def word2vec_model(sentences, vector_size=100, min_count=5, window=5, negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a gensim Word2Vec model.
    
    Args:
        sentences (list): A list of sentences (list of lists of strings) 
                          to be trained on.
        vector_size (int): The dimensionality of the embedding layer.
        min_count (int): The minimum number of occurrences of a word 
                         for use in training.
        window (int): The maximum distance between the current and 
                      predicted word within a sentence.
        negative (int): The size of negative sampling.
        cbow (bool): A boolean to determine the training type. 
                     True is for CBOW, False is for Skip-gram.
        epochs (int): The number of iterations to train over.
        seed (int): The seed for the random number generator.
        workers (int): The number of worker threads to train the model.
        
    Returns:
        gensim.models.word2vec.Word2Vec: The trained Word2Vec model.
    """
    
    # 1. Create the Word2Vec model instance using the full namespace.
    # sg=0 for CBOW (default), sg=1 for Skip-gram.
    model = gensim.models.Word2Vec(
        sentences=None,  # Pass None initially
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=0 if cbow else 1,
        seed=seed,
        workers=workers
    )

    # 2. Build the vocabulary
    model.build_vocab(sentences=sentences)

    # 3. Train the model
    total_examples = model.corpus_count
    
    model.train(
        corpus_iterable=sentences,
        total_examples=total_examples,
        epochs=epochs
    )

    return model
