#!/usr/bin/env python3
"""Performs Gaussian Mixture Modeling using sklearn"""
import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset using sklearn

    Parameters:
    - X: np.ndarray of shape (n, d), dataset
    - k: int, number of clusters

    Returns:
    - pi: np.ndarray of shape (k,), cluster priors
    - m: np.ndarray of shape (k, d), centroid means
    - S: np.ndarray of shape (k, d, d), covariance matrices
    - clss: np.ndarray of shape (n,), cluster assignments
    - bic: float, Bayesian Information Criterion value
    """
    gmm_model = sklearn.mixture.GaussianMixture(n_components=k)
    gmm_model.fit(X)

    pi = gmm_model.weights_
    m = gmm_model.means_
    S = gmm_model.covariances_
    clss = gmm_model.predict(X)
    bic = gmm_model.bic(X)

    return pi, m, S, clss, bic
