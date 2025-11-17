#!/usr/bin/env python3
"""
Module defining the Dataset class for machine translation.
"""
import tensorflow_datasets as tfds
import transformers
# Removed: import numpy as np


class Dataset:
    """
    Loads and preps a dataset for machine translation.
    """

    def __init__(self):
        """
        Constructor for the Dataset class.
        # ... (rest of __init__ as before)
        """
        # Load the dataset
        data_splits, info = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True,
            with_info=True
        )
        self.data_train = data_splits[0]
        self.data_valid = data_splits[1]

        # Create the tokenizers
        self.vocab_size = 2**13 # Added this for encode method
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)


    def tokenize_dataset(self, data):
        """
        # ... (tokenize_dataset method as before)
        """
        # Portuguese Tokenizer: neuralmind/bert-base-portuguese-cased
        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )

        # English Tokenizer: bert-base-uncased
        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased'
        )

        # Generator function for the Portuguese text
        def pt_generator():
            """Yields Portuguese sentences from the dataset."""
            for pt, _ in data.as_numpy_iterator():
                yield pt.decode('utf-8')

        # Generator function for the English text
        def en_generator():
            """Yields English sentences from the dataset."""
            for _, en in data.as_numpy_iterator():
                yield en.decode('utf-8')

        # Train the Portuguese tokenizer
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            text_iterator=pt_generator(),
            vocab_size=self.vocab_size,
            initial_alphabet=list(tokenizer_pt.get_vocab().keys())
        )

        # Train the English tokenizer
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            text_iterator=en_generator(),
            vocab_size=self.vocab_size,
            initial_alphabet=list(tokenizer_en.get_vocab().keys())
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens.
        
        # ... (rest of docstring)
        """
        # Decode tf.Tensor to string
        pt_string = pt.numpy().decode('utf-8')
        en_string = en.numpy().decode('utf-8')

        # Encode sentences
        pt_tokens = self.tokenizer_pt.encode(
            pt_string, 
            max_length=None, 
            truncation=False,
            padding=False
        ).as_numpy_array()
        
        en_tokens = self.tokenizer_en.encode(
            en_string, 
            max_length=None, 
            truncation=False,
            padding=False
        ).as_numpy_array()

        # Custom token indices
        pt_tokens[0] = self.vocab_size
        en_tokens[0] = self.vocab_size
        
        pt_tokens[-1] = self.vocab_size + 1
        en_tokens[-1] = self.vocab_size + 1

        return pt_tokens, en_tokens
