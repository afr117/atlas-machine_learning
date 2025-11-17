#!/usr/bin/env python3
"""
Module defining the Dataset class for machine translation.
"""
import tensorflow_datasets as tfds
from transformers import BertTokenizerFast

class Dataset:
    """
    Loads and preps a dataset for machine translation.
    """

    def __init__(self):
        """
        Constructor for the Dataset class.

        Creates the instance attributes:
        - data_train: the tf.data.Dataset train split (as_supervised).
        - data_valid: the tf.data.Dataset validate split (as_supervised).
        - tokenizer_pt: the Portuguese tokenizer created from the training set.
        - tokenizer_en: the English tokenizer created from the training set.
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
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

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
        # Instantiate the tokenizer (fast version is used by default)
        tokenizer_pt = BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased',
            max_model_input_sizes=512 # Default value
        )
        
        # English Tokenizer: bert-base-uncased
        tokenizer_en = BertTokenizerFast.from_pretrained(
            'bert-base-uncased',
            max_model_input_sizes=512 # Default value
        )

        # Generator function for the Portuguese text
        def pt_generator():
            """Yields Portuguese sentences from the dataset."""
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        # Generator function for the English text
        def en_generator():
            """Yields English sentences from the dataset."""
            for _, en in data:
                yield en.numpy().decode('utf-8')

        # Train the Portuguese tokenizer
        # The training set is used to determine the vocabulary of the tokenizer.
        # We use the train_new_from_iterator method to build the vocab.
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            text_iterator=pt_generator(),
            vocab_size=2**13,
            initial_alphabet=tokenizer_pt.get_vocab().keys()
        )

        # Train the English tokenizer
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            text_iterator=en_generator(),
            vocab_size=2**13,
            initial_alphabet=tokenizer_en.get_vocab().keys()
        )

        return tokenizer_pt, tokenizer_en
