"""PCA utilities with standard scaling and optional visualization."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import gaussian_kde


def EigenValueDecomposition(dataset, alg=None, title=None, visualize_ratio='no'):
    scaler = StandardScaler()
    pca = PCA()
    pipeline = make_pipeline(scaler, pca)
    pipeline.fit(dataset)

    if visualize_ratio == 'yes':
        features = range(pca.n_components_)
        plt.bar(features, pca.explained_variance_)
        plt.xlabel('features')
        plt.ylabel('variance')
        if alg is not None and title is not None:
            plt.title(str(alg).upper() + ' Variance - ' + str(title))
        plt.xticks(features)
        plt.show()
        plt.close()
    return pca.explained_variance_ratio_, pca.components_


def DimensionReduction(dataset, n_components=3, alg=None, title=None):
    scaler = StandardScaler()
    pca = PCA(n_components=n_components)
    pipeline = make_pipeline(scaler, pca)
    pca_results = pipeline.fit_transform(dataset)
    return pca_results


def Visualize2D(pca_results_2D, title=None):
    fig, ax = plt.subplots()
    xf = pca_results_2D[:, 0]
    yf = pca_results_2D[:, 1]

    xy = np.vstack([xf, yf])
    z = gaussian_kde(xy)(xy)

    plt.scatter(xf, yf, c=z, s=50, edgecolors='none')
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.show()
    plt.close()


def Visualize3D(pca_results_3D, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xf = pca_results_3D[:, 0]
    yf = pca_results_3D[:, 1]
    zf = pca_results_3D[:, 2]
    ax.scatter(xf, yf, zf, s=0.1, color='blue')
    if title is not None:
        plt.title(title)
    plt.show()
    plt.close()

__all__ = [
    'EigenValueDecomposition',
    'DimensionReduction',
    'Visualize2D',
    'Visualize3D',
]
