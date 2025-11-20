#!/usr/bin/env python3
"""
Implements the final multi-reference Question Answering loop.
This function uses semantic search to find the best document from a corpus
and then uses a BERT model to extract the exact answer snippet from that document.
"""
import sys

# --- Import Functions from Previous Tasks ---

# 1. Answer Extraction Function (from 2-qa.py)
# This function takes (question, reference_text) and returns the answer snippet or None.
try:
    qa_module = __import__('2-qa')
    question_answer_extract = qa_module.question_answer
except ImportError:
    print("Error: Could not import question_answer function from 2-qa.py. Ensure the file is correct.", file=sys.stderr)
    sys.exit(1)

# 2. Semantic Search Function (from 3-semantic-search.py)
# This function takes (corpus_path, sentence) and returns the full text of the best document.
try:
    search_module = __import__('3-semantic_search')
    semantic_search = search_module.semantic_search
except ImportError:
    print("Error: Could not import semantic_search function from 3-semantic_search.py. Ensure the file is correct.", file=sys.stderr)
    sys.exit(1)


def question_answer(corpus_path):
    """
    Implements a loop to answer questions by performing a two-step process:
    1. Search the corpus for the most relevant document (semantic search).
    2. Extract the precise answer from that document (BERT QA model).

    Args:
        corpus_path (str): The path to the directory containing the reference documents.
    """
    exit_commands = ['exit', 'quit', 'goodbye', 'bye']
    not_found_response = "Sorry, I do not understand your question."

    # A friendly initial message
    print("--- Multi-Reference QA Bot Initialized (Type 'exit' to quit) ---")

    while True:
        # 1. Get user input
        user_input = input("Q: ")
        
        # 2. Check for exit commands (case-insensitive)
        if user_input.lower() in exit_commands:
            print("A: Goodbye")
            break
        
        # 3. Step 1: Document Retrieval (Semantic Search)
        # Find the full text of the single best reference document for the query.
        reference = semantic_search(corpus_path, user_input)
        
        if reference is None:
            # This handles cases where corpus is empty or the search model failed
            print(f"A: {not_found_response}")
            continue

        # 4. Step 2: Answer Extraction (BERT QA Model)
        # Use the imported extraction function to get the snippet from the retrieved document.
        answer = question_answer_extract(user_input, reference)
        
        # 5. Print the result
        if answer is None:
            # This means either the semantic search failed to find a truly relevant document,
            # OR the QA extraction model could not find a valid snippet in the text it was given.
            print(f"A: {not_found_response}")
        else:
            # Print the extracted answer snippet
            print(f"A: {answer}")


if __name__ == '__main__':
    # The driver script 4-main.py is expected to call question_answer('ZendeskArticles')
    pass
