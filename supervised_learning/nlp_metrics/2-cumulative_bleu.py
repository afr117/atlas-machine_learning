#!/usr/bin/env python3
"""
Calculate cumulative n-gram BLEU score for a sentence
"""
import math


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a proposed sentence.

    Args:
        references (list of list of str): reference translations.
        sentence (list of str): proposed sentence to evaluate.
        n (int): highest n-gram order to use.

    Returns:
        float: cumulative BLEU score.
    """

    # Function to compute clipped precision for a given n-gram size
    def n_gram_precision(n_size):
        # Candidate n-grams
        cand_ngrams = [
            tuple(sentence[i:i + n_size])
            for i in range(len(sentence) - n_size + 1)
        ]
        if len(cand_ngrams) == 0:
            return 0.0

        cand_counts = {}
        for ng in cand_ngrams:
            cand_counts[ng] = cand_counts.get(ng, 0) + 1

        max_ref_counts = {}
        for ref in references:
            ref_ngrams = [
                tuple(ref[i:i + n_size])
                for i in range(len(ref) - n_size + 1)
            ]
            ref_counts = {}
            for ng in ref_ngrams:
                ref_counts[ng] = ref_counts.get(ng, 0) + 1

            for ng, c in ref_counts.items():
                if ng not in max_ref_counts or c > max_ref_counts[ng]:
                    max_ref_counts[ng] = c

        clipped = sum(
            min(c, max_ref_counts.get(ng, 0))
            for ng, c in cand_counts.items()
        )
        return clipped / len(cand_ngrams)

    # Compute weighted geometric mean of precisions
    weights = [1 / n] * n  # uniform weights
    precisions = [n_gram_precision(i) for i in range(1, n + 1)]

    # Avoid log(0) → If any precision = 0, BLEU = 0
    if any(p == 0 for p in precisions):
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(
            w * math.log(p) for w, p in zip(weights, precisions)
        ))

    # Brevity Penalty (BP)
    ref_lens = [len(ref) for ref in references]
    len_c = len(sentence)
    closest_ref_len = min(
        ref_lens, key=lambda rl: (abs(rl - len_c), rl)
    )

    if len_c > closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - closest_ref_len / len_c)

    return bp * geo_mean
