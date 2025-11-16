#!/usr/bin/env python3
"""
Calculate the n-gram BLEU score for a sentence
"""
import math


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a model-proposed sentence.

    Args:
        references (list of list of str): reference translations.
        sentence (list of str): proposed sentence to evaluate.
        n (int): size of the n-gram to use.

    Returns:
        float: n-gram BLEU score.
    """
    # Candidate n-grams
    cand_ngrams = [
        tuple(sentence[i:i + n]) for i in range(len(sentence) - n + 1)
    ]
    len_cand = len(cand_ngrams)
    if len_cand <= 0:
        return 0.0

    # Count candidate n-grams
    cand_counts = {}
    for ng in cand_ngrams:
        cand_counts[ng] = cand_counts.get(ng, 0) + 1

    # Max reference counts (for clipping)
    max_ref_counts = {}
    for ref in references:
        ref_ngrams = [
            tuple(ref[i:i + n]) for i in range(len(ref) - n + 1)
        ]
        ref_counts = {}
        for ng in ref_ngrams:
            ref_counts[ng] = ref_counts.get(ng, 0) + 1
        for ng, c in ref_counts.items():
            if ng not in max_ref_counts or c > max_ref_counts[ng]:
                max_ref_counts[ng] = c

    # Clipped precision
    clipped_matches = 0
    for ng, c in cand_counts.items():
        clipped_matches += min(c, max_ref_counts.get(ng, 0))

    precision = clipped_matches / len_cand

    # Brevity Penalty (BP)
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens,
                          key=lambda rl: (abs(rl - len(sentence)), rl))

    if len(sentence) > closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - closest_ref_len / len(sentence))

    return bp * precision
