#!/usr/bin/env python3
"""
Function to create an interactive loop that answers questions 
from a reference text using a BERT Question Answering model.
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
    model = hub.KerasLayer(
        'https://tfhub.dev/see--/bert-uncased-tf2-qa/1',
        trainable=False
    )

    # --- 1. Prepare Input ---
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
    # Use a list of inputs for positional arguments to satisfy the SavedModel signature.
    result = model(
        [input_word_ids, input_mask, input_type_ids]
    )
    
    # Squeeze to remove batch dimension (1, seq_len) -> (seq_len)
    start_logits = tf.squeeze(result[0])
    end_logits = tf.squeeze(result[1])

    # --- 3. Post-processing and Answer Extraction (Robust Span Search) ---
    
    tokens = tokenizer.convert_ids_to_tokens(input_word_ids.numpy()[0])
    
    start_probs = start_logits.numpy()
    end_probs = end_logits.numpy()
    type_ids = input_type_ids.numpy()[0]
    
    best_score = -1e10 
    best_start = -1
    best_end = -1
    max_answer_length = 30 
    
    try:
        context_start_index = np.where(type_ids == 1)[0][0]
    except IndexError:
        return None

    for i in range(context_start_index, len(start_probs)):
        if tokens[i] == '[SEP]':
             break
        if type_ids[i] != 1:
            continue
            
        for j in range(i, len(end_probs)):
            if tokens[j] == '[SEP]' or (j - i + 1) > max_answer_length:
                 break
            if type_ids[j] != 1:
                continue
                 
            score = start_probs[i] + end_probs[j]
            
            if score > best_score:
                best_score = score
                best_start = i
                best_end = j

    if best_start == -1 or best_end == -1:
        return None

    answer_tokens = tokens[best_start:best_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    
    if not answer.strip():
        return None
        
    return answer


def answer_loop(reference):
    """
    Answers questions from the user based on a reference text.

    Args:
        reference (str): The reference document from which to find answers.
    """
    exit_commands = ['exit', 'quit', 'goodbye', 'bye']
    not_found_response = "Sorry, I do not understand your question."

    while True:
        # Prompt the user for input
        user_input = input("Q: ")
        
        # Check for exit commands (case-insensitive)
        if user_input.lower() in exit_commands:
            print("A: Goodbye")
            break
        
        # Get the answer from the reference text
        answer = question_answer(user_input, reference)
        
        # Print the answer or the "not found" response
        if answer is None:
            print(f"A: {not_found_response}")
        else:
            print(f"A: {answer}")


if __name__ == '__main__':
    # This block is for testing the function directly if needed,
    # but the task specifies using a 2-main.py driver.
    pass
