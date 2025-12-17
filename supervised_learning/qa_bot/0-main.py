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
    # Load the pre-trained tokenizer from the transformers library
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )

    # Load the QA model from TensorFlow Hub
    model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')

    # Tokenize the input (question and reference)
    # This automatically adds [CLS] and [SEP] tokens
    inputs = tokenizer.encode_plus(question, reference, return_tensors='tf')

    input_word_ids = inputs['input_ids']
    input_mask = inputs['attention_mask']
    input_type_ids = inputs['token_type_ids']

    # Get the model outputs (start and end logits)
    outputs = model([input_word_ids, input_mask, input_type_ids])
    
    # outputs[0] are the start_logits, outputs[1] are the end_logits
    short_start = tf.argmax(outputs[0], axis=1).numpy()[0]
    short_end = tf.argmax(outputs[1], axis=1).numpy()[0]

    # Convert the token IDs back to a string
    # We select the tokens between the predicted start and end indices
    answer_tokens = input_word_ids[0, short_start:short_end + 1]
    answer = tokenizer.decode(answer_tokens)

    # If no valid answer tokens were found, return None
    if not answer or answer.strip() == "":
        return None

    return answer
