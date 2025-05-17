#!/usr/bin/env python3
"""Performs K-means clustering using sklearn"""
import numpy as np
from sklearn.cluster import KMeans


def kmeans(X, k):
    """
    Performs K-means on a dataset using sklearn

    Parameters:
    - X: np.ndarray of shape (n, d), dataset
    - k: int, number of clusters

    Returns:
    - C: np.ndarray of shape (k, d), centroid coordinates
    - clss: np.ndarray of shape (n,), index of the cluster each point belongs to
    """
    kmeans_model = KMeans(n_clusters=k, n_init='auto')
    kmeans_model.fit(X)
    C = kmeans_model.cluster_centers_
    clss = kmeans_model.labels_

    return C, clss
