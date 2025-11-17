#!/usr/bin/env python3
"""
Dataset class for machine translation using TED HRLR pt-en and BERT tokenizers
"""

import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Loads and prepares a dataset for machine translation.

    Attributes:
        data_train (tf.data.Dataset): training split of pt-en dataset.
        data_valid (tf.data.Dataset): validation split of pt-en dataset.
        tokenizer_pt: Portuguese tokenizer (BertTokenizerFast).
        tokenizer_en: English tokenizer (BertTokenizerFast).
    """

    def __init__(self):
        """
        Initializes the Dataset instance by loading the train and validation
        splits and creating the Portuguese and English tokenizers.
        """
        self.data_train = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split="train",
            as_supervised=True,
        )
        self.data_valid = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split="validation",
            as_supervised=True,
        )

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset using pretrained models.

        Args:
            data (tf.data.Dataset): dataset of (pt, en) sentence pairs.
                This argument is kept for API compatibility, but the
                tokenizers are loaded from pretrained BERT models.

        Returns:
            tuple: (tokenizer_pt, tokenizer_en)
                tokenizer_pt: Portuguese tokenizer.
                tokenizer_en: English tokenizer.
        """
        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            "bert-base-uncased"
        )
        return tokenizer_pt, tokenizer_en
