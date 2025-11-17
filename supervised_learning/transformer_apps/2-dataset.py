#!/usr/bin/env python3
"""
Module defining the Dataset class for machine translation.
"""
import tensorflow_datasets as tfds
import transformers
import tensorflow as tf
import numpy as np


class Dataset:
    """
    Loads and preps a dataset for machine translation, and encodes it into tokens.
    """

    def __init__(self):
        """
        Constructor for the Dataset class.

        Creates and tokenizes the instance attributes:
        - data_train: the tf.data.Dataset train split, tokenized.
        - data_valid: the tf.data.Dataset validate split, tokenized.
        - tokenizer_pt: the Portuguese tokenizer created from the training set.
        - tokenizer_en: the English tokenizer created from the training set.
        """
        # Load the dataset (raw text)
        data_splits, info = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True,
            with_info=True
        )
        data_train_raw = data_splits[0]
        data_valid_raw = data_splits[1]

        # Create the tokenizers
        self.vocab_size = 2**13
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(data_train_raw)

        # Tokenize the datasets using the TensorFlow wrapper
        self.data_train = data_train_raw.map(
            lambda pt, en: tf.py_function(
                self.tf_encode,
                [pt, en],
                [tf.int64, tf.int64]
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        self.data_valid = data_valid_raw.map(
            lambda pt, en: tf.py_function(
                self.tf_encode,
                [pt, en],
                [tf.int64, tf.int64]
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for our dataset.

        Args:
            data: A tf.data.Dataset whose examples are formatted as a tuple (pt, en).
                pt is the tf.Tensor containing the Portuguese sentence.
                en is the tf.Tensor containing the corresponding English sentence.

        Returns:
            tokenizer_pt: The Portuguese tokenizer (BertTokenizerFast).
            tokenizer_en: The English tokenizer (BertTokenizerFast).
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

        Args:
            pt: tf.Tensor containing the Portuguese sentence.
            en: tf.Tensor containing the corresponding English sentence.

        Returns:
            pt_tokens: A np.ndarray containing the Portuguese tokens.
            en_tokens: A np.ndarray containing the English tokens.
        """
        # Decode tf.Tensor (as numpy byte string) to Python string
        pt_string = pt.numpy().decode('utf-8')
        en_string = en.numpy().decode('utf-8')

        # Encode Portuguese sentence using its tokenizer
        pt_tokens = self.tokenizer_pt.encode(
            pt_string, 
            max_length=None, 
            truncation=False,
            padding=False
        ).as_numpy_array()
        
        # Encode English sentence using its tokenizer
        en_tokens = self.tokenizer_en.encode(
            en_string, 
            max_length=None, 
            truncation=False,
            padding=False
        ).as_numpy_array()

        # Replace the first token ([CLS]) with the SOS token (vocab_size)
        pt_tokens[0] = self.vocab_size
        en_tokens[0] = self.vocab_size
        
        # Replace the last token ([SEP]) with the EOS token (vocab_size + 1)
        pt_tokens[-1] = self.vocab_size + 1
        en_tokens[-1] = self.vocab_size + 1

        return pt_tokens, en_tokens
    
    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for the encode instance method.

        Args:
            pt: tf.Tensor containing the Portuguese sentence.
            en: tf.Tensor containing the corresponding English sentence.

        Returns:
            pt_tokens: A tf.Tensor containing the Portuguese tokens.
            en_tokens: A tf.Tensor containing the English tokens.
        """
        pt_tokens, en_tokens = tf.py_function(
            self.encode, 
            [pt, en], 
            [tf.int64, tf.int64]
        )
        
        # Manually set shapes as tf.py_function loses shape information
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])
        
        return pt_tokens, en_tokens
