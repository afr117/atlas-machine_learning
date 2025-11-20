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
    # Final Fix: Use a list of inputs, not a tuple, to match the positional 
    # arguments (inputs/0, inputs/1, inputs/2) explicitly as a list structure.
    result = model(
        [input_word_ids, input_mask, input_type_ids]
    )
    
    # Squeeze to remove batch dimension (1, seq_len) -> (seq_len)
    start_logits = tf.squeeze(result[0])
    end_logits = tf.squeeze(result[1])

    # --- 3. Post-processing and Answer Extraction ---
    
    # Get numpy arrays for easier manipulation
    start_probs = start_logits.numpy()
    end_probs = end_logits.numpy()
    type_ids = input_type_ids.numpy()[0]
    
    # Initialize best score and indices
    best_score = -1e10 # Very low number
    best_start = -1
    best_end = -1
    
    # Search for the best span (i, j) where i <= j and i is in the context (type_ids[i] == 1)
    max_answer_length = 30 # Standard limit, preventing excessively long/nonsensical answers
    
    # We start searching from the beginning of the context (type_id 1)
    context_start_index = np.where(type_ids == 1)[0][0]

    for i in range(context_start_index, len(start_probs)):
        # Stop if we hit the padding tokens or exceed max_answer_length from the end
        if tokens[i] == '[SEP]' or i > best_start + max_answer_length:
             break
        
        # Ensure the start token is in the context
        if type_ids[i] != 1:
            continue
            
        for j in range(i, len(end_probs)):
            # Ensure the end token is still in the context and limit length
            if type_ids[j] != 1 or (j - i + 1) > max_answer_length:
                 break
                 
            score = start_probs[i] + end_probs[j]
            
            if score > best_score:
                best_score = score
                best_start = i
                best_end = j

    # Check for a valid answer span based on the search
    if best_start == -1 or best_end == -1:
        return None

    # The actual tokens corresponding to the input indices
    tokens = tokenizer.convert_ids_to_tokens(input_word_ids.numpy()[0])
    
    # Extract the answer tokens and convert back to a string
    answer_tokens = tokens[best_start:best_end + 1]

    # Join the tokens and clean up the BERT specific '##' subword notation
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    
    # Final check for empty result 
    if not answer.strip():
        return None
        
    return answer
