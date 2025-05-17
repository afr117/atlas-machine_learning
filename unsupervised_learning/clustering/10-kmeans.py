#!/usr/bin/env python3
"""Performs K-means clustering using sklearn"""
import sklearn.cluster


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
    model = sklearn.cluster.KMeans(n_clusters=k, n_init='auto')
    model.fit(X)
    return model.cluster_centers_, model.labels_
