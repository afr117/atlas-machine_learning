#!/usr/bin/env python3
"""
Determines if gradient descent should stop early based on validation cost.
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early.

    Args:
        cost (float): Current validation cost of the neural network.
        opt_cost (float): Lowest recorded validation cost of the neural network.
        threshold (float): Threshold used for early stopping.
        patience (int): Patience count used for early stopping.
        count (int): Count of how long the threshold has not been met.
  
    Returns:
        tuple: (bool, int) where bool indicates if
        training should stop, and int is the updated count.
    """
    if (opt_cost - cost) > threshold:
        return False, 0
    else:
        count += 1
        if count >= patience:
            return True, count
        return False, count
