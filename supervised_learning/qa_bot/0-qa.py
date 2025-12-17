#!/usr/bin/env python3
"""
Module for Question Answering using BERT and TensorFlow Hub
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question.

    Args:
        question (str): The question to answer.
        reference (str): The document containing the answer.

    Returns:
        str: A string containing the answer, or None if no answer is found.
    """
    # Load the pre-trained tokenizer
    model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Load the QA model from TensorFlow Hub
    model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')

    # Tokenize the input
    inputs = tokenizer.encode_plus(question, reference, return_tensors='tf')

    input_word_ids = inputs['input_ids']
    input_mask = inputs['attention_mask']
    input_type_ids = inputs['token_type_ids']

    # Get model outputs (start_logits are index 0, end_logits are index 1)
    outputs = model([input_word_ids, input_mask, input_type_ids])
    start_logits = outputs[0]
    end_logits = outputs[1]

    # Find start and end indices, ignoring the [CLS] token at index 0
    # We add 1 to the result because argmax on [1:] returns indices starting at 0
    short_start = tf.argmax(start_logits[:, 1:], axis=1).numpy()[0] + 1
    short_end = tf.argmax(end_logits[:, 1:], axis=1).numpy()[0] + 1

    # Convert the IDs to tokens
    all_tokens = tokenizer.convert_ids_to_tokens(input_word_ids[0])

    # Extract the answer tokens
    answer_tokens = all_tokens[short_start:short_end + 1]

    # Convert tokens to a clean string
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    if not answer or answer.strip() == "" or answer == "[CLS]":
        return None

    return answer
