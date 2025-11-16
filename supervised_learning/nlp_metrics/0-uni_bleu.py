#!/usr/bin/env python3
"""
Unigram BLEU score
"""


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a proposed sentence.

    Args:
        references (list of list of str): reference translations, where each
            reference is a list of words.
        sentence (list of str): model-proposed sentence as a list of words.

    Returns:
        float: the unigram BLEU score.
    """
    # If candidate is empty, BLEU is 0 by definition
    len_cand = len(sentence)
    if len_cand == 0:
        return 0.0

    # ----- 1. Compute clipped unigram precision -----
    # Count unigrams in candidate
    cand_counts = {}
    for w in sentence:
        cand_counts[w] = cand_counts.get(w, 0) + 1

    # For each word, find max reference count (for clipping)
    max_ref_counts = {}
    for ref in references:
        ref_counts = {}
        for w in ref:
            ref_counts[w] = ref_counts.get(w, 0) + 1
        for w, c in ref_counts.items():
            if w not in max_ref_counts or c > max_ref_counts[w]:
                max_ref_counts[w] = c

    # Clipped count: min(count_in_candidate, max_ref_count_over_refs)
    clipped_matches = 0
    for w, c in cand_counts.items():
        clipped_matches += min(c, max_ref_counts.get(w, 0))

    # Unigram precision
    precision = clipped_matches / len_cand

    # ----- 2. Compute brevity penalty (BP) -----
    # Choose reference length closest to candidate length
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda rl: (abs(rl - len_cand), rl))

    if len_cand > closest_ref_len:
        bp = 1.0
    else:
        bp = pow(2.718281828459045, 1 - closest_ref_len / len_cand)

    # ----- 3. Unigram BLEU -----
    bleu = bp * precision
    return bleu
