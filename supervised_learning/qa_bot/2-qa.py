#!/usr/bin/env python3
"""
Module for an integrated QA loop using BERT
"""
import tensorflow as tf


def answer_loop(reference):
    """
    Answers questions from a reference text using a BERT model loop

    Args:
        reference (str): The text containing the information to search
    """
    # Import the function from task 0
    question_answer = __import__('0-qa').question_answer
    exit_commands = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        # Prompt the user
        user_input = input("Q: ")

        # Check for exit commands (case-insensitive)
        if user_input.lower() in exit_commands:
            print("A: Goodbye")
            break

        # Get the answer from the BERT model
        answer = question_answer(user_input, reference)

        # Handle cases where the model finds no answer
        if answer is None or answer.strip() == "":
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: {}".format(answer))


