#!/usr/bin/env python3
import gensim

def word2vec_model(sentences, vector_size=100, min_count=5, window=5, negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a gensim Word2Vec model.
    # ... (docstring omitted for brevity)
    """
    
    # 1. Create the Word2Vec model instance.
    # NOTE: Omiting 'sentences' in the constructor to perform build_vocab later.
    model = gensim.models.Word2Vec(
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=0 if cbow else 1,
        seed=seed,
        workers=workers
    )

    # 2. Build the vocabulary
    # The vocabulary needs to be built before training can start.
    model.build_vocab(sentences=sentences)

    # 3. Train the model
    # We pass the sentences and explicitly specify the total_examples and epochs.
    total_examples = model.corpus_count
    
    model.train(
        corpus_iterable=sentences,
        total_examples=total_examples,
        epochs=epochs
    )

    return model
