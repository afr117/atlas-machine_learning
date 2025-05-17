#!/usr/bin/env python3
"""Performs agglomerative clustering and displays a dendrogram"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering with Ward linkage

    Parameters:
    - X: np.ndarray of shape (n, d), dataset
    - dist: float, maximum cophenetic distance for clusters

    Returns:
    - clss: np.ndarray of shape (n,), cluster indices for each point
    """
    linkage_matrix = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(linkage_matrix, dist, criterion='distance')

    plt.figure()
    scipy.cluster.hierarchy.dendrogram(linkage_matrix, color_threshold=dist)
    plt.axhline(y=dist, c='k', ls='--')
    plt.show()

    return clss
