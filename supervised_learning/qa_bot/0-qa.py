#!/usr/bin/env python3
"""
Function to find a snippet of text within a reference document to answer a question.
Uses the BERT QA model fine-tuned on SQuAD.
"""
import tensorflow as tf
import tensorflow_hub as hub
import transformers
import numpy as np


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question.

    Args:
        question (str): A string containing the question to answer.
        reference (str): A string containing the reference document.

    Returns:
        str: A string containing the answer, or None if no answer is found.
    """
    # Use the specified pre-trained tokenizer
    tokenizer = transformers.BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )

    # Use the specified TensorFlow Hub model
    # Note: Hub models are loaded via a KerasLayer
    model = hub.KerasLayer(
        'https://tfhub.dev/see--/bert-uncased-tf2-qa/1',
        trainable=False
    )

    # --- 1. Prepare Input ---
    # Tokenize the question and reference together.
    encoded = tokenizer.encode_plus(
        question,
        reference,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='tf'
    )

    input_word_ids = encoded['input_ids']
    input_mask = encoded['attention_mask']
    input_type_ids = encoded['token_type_ids']

    # --- 2. Model Inference ---
    # The SavedModel requires named dictionary inputs corresponding to the
    # input names defined in the model (Options 3 & 4 of the error message).
    result = model({
        'input_word_ids': input_word_ids,
        'input_mask': input_mask,
        'input_type_ids': input_type_ids
    })
    
    start_logits = result[0]
    end_logits = result[1]

    # --- 3. Post-processing and Answer Extraction ---
    
    # Get the predicted start and end index by finding the max logit
    # Squeeze to remove the batch dimension (shape is 1, seq_len)
    start_index = tf.argmax(start_logits, axis=-1).numpy()[0]
    end_index = tf.argmax(end_logits, axis=-1).numpy()[0]

    # The actual tokens corresponding to the input indices
    tokens = tokenizer.convert_ids_to_tokens(input_word_ids.numpy()[0])
    
    # Check for a valid answer span:
    # 1. Start index must be before or at the end index.
    # 2. Start index must point to a token in the reference document (type ID 1).
    if start_index > end_index or input_type_ids.numpy()[0][start_index] != 1:
        return None

    # Extract the answer tokens and convert back to a string
    answer_tokens = tokens[start_index:end_index + 1]

    # Join the tokens and clean up the BERT specific '##' subword notation
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    
    # Final check for empty result 
    if not answer.strip():
        return None
        
    return answer
