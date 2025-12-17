#!/usr/bin/env python3
"""
Module for multi-reference Question Answering
"""
import tensorflow as tf


def question_answer(corpus_path):
    """
    Answers questions from multiple reference texts

    Args:
        corpus_path (str): Path to the corpus of reference documents
    """
    # Import relevant functions from previous tasks
    # We use __import__ because filenames start with numbers
    semantic_search = __import__('3-semantic_search').semantic_search
    qa_func = __import__('0-qa').question_answer
    
    exit_commands = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        # Prompt the user
        user_input = input("Q: ")

        # Check for exit commands (case-insensitive)
        if user_input.lower() in exit_commands:
            print("A: Goodbye")
            break

        # 1. Find the most relevant document in the folder
        reference = semantic_search(corpus_path, user_input)

        # 2. Find the specific answer within that document
        answer = qa_func(user_input, reference)

        # 3. Handle cases where no answer is found
        if answer is None or answer.strip() == "" or answer == "[CLS]":
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: {}".format(answer))
