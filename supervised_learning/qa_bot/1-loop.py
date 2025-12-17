#!/usr/bin/env python3
"""
Module for a simple interactive loop
"""


def question_loop():
    """
    Continually prompts the user for input and handles exit commands
    """
    exit_commands = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        # Prompt the user
        user_input = input("Q: ")

        # Check for exit commands (case-insensitive)
        if user_input.lower() in exit_commands:
            print("A: Goodbye")
            break

        # For Task 1, we just print "A: " and continue
        print("A: ")


if __name__ == "__main__":
    question_loop()
