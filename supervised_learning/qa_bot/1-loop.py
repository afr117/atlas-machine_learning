#!/usr/bin/env python3
"""
Script to create a simple interactive loop for a QA bot.
"""

def main():
    """
    Runs the interactive Q&A loop.
    """
    exit_commands = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        # Prompt the user for input
        user_input = input("Q: ")
        
        # Convert input to lowercase for case-insensitive check
        if user_input.lower() in exit_commands:
            print("A: Goodbye")
            break
        
        # In this task, we just print A: and wait for the next input
        # In a later task, the answer generation function will go here.
        print("A:")

if __name__ == '__main__':
    main()
