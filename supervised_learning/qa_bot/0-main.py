#!/usr/bin/env python3

question_answer = __import__('0-qa').question_answer # Imports the function

with open('ZendeskArticles/PeerLearningDays.md') as f:
    reference = f.read()

# The print statement executes the function and displays its return value.
print(question_answer('When are PLDs?', reference))
