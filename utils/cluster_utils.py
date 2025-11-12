"""Clustering helpers (KMeans + silhouette + optional visualization)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def kmeans_clustering(data, num_clusters, n_init=10):
    kmeans = KMeans(n_clusters=num_clusters, n_init=n_init)
    kmeans.fit(data)
    return np.asarray(kmeans.labels_)


def calculate_silhouette(data, labels):
    return float(silhouette_score(data, labels))


def visualize_clusters_2D(data, labels, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[:, 0], data[:, 1], c=labels.astype(float), edgecolor='k')
    if title is not None:
        plt.title(title)
    plt.show()
    plt.close()


def visualize_clusters_3D(data, labels, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels.astype(float), edgecolor='k')
    if title is not None:
        plt.title(title)
    plt.show()
    plt.close()

__all__ = [
    'kmeans_clustering',
    'calculate_silhouette',
    'visualize_clusters_2D',
    'visualize_clusters_3D',
]
