import os
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except Exception:  # pragma: no cover
    Axes3D = None  # type: ignore

try:
    from scipy.stats import gaussian_kde
except Exception:  # optional; density plot falls back if missing
    gaussian_kde = None  # type: ignore


def _save_or_show(path: Optional[str] = None, *, dpi: int = 150, show: bool = False):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=dpi)
        plt.close()
    elif show:
        plt.show()
        plt.close()
    else:
        plt.close()


def plot_evr(evr_vals: Sequence[float], path: Optional[str] = None, *, show: bool = False):
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(evr_vals) + 1), evr_vals, color="#4C78A8")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Explained Variance Ratio")
    plt.tight_layout()
    _save_or_show(path, show=show)


def scatter_2d(points: np.ndarray,
               labels: Optional[np.ndarray] = None,
               path: Optional[str] = None,
               *,
               title: Optional[str] = None,
               cmap: str = "coolwarm",
               s: float = 4,
               alpha: float = 0.8,
               density: bool = False,
               show: bool = False):
    plt.figure(figsize=(7, 6))
    if density and gaussian_kde is not None and labels is None:
        xf = points[:, 0]
        yf = points[:, 1]
        xy = np.vstack([xf, yf])
        z = gaussian_kde(xy)(xy)
        sc = plt.scatter(xf, yf, c=z, s=s, alpha=alpha, cmap="viridis")
        plt.colorbar(sc)
    else:
        sc = plt.scatter(points[:, 0], points[:, 1], c=labels, s=s, cmap=cmap, alpha=alpha)
        if labels is not None:
            plt.colorbar(sc)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    if title:
        plt.title(title)
    plt.tight_layout()
    _save_or_show(path, show=show)


def scatter_3d(points3d: np.ndarray,
               labels: Optional[np.ndarray] = None,
               path: Optional[str] = None,
               *,
               title: Optional[str] = None,
               cmap: str = "tab20",
               s: float = 1,
               alpha: float = 0.6,
               show: bool = False):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], c=labels, cmap=cmap, s=s, alpha=alpha)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    if title:
        ax.set_title(title)
    if labels is not None:
        plt.colorbar(sc, ax=ax)
    plt.tight_layout()
    _save_or_show(path, show=show)


def plot_elbow_curve(k_vals: Sequence[int], inertias: Sequence[float], path: Optional[str] = None, *, show: bool = False):
    plt.figure(figsize=(7, 5))
    plt.plot(k_vals, inertias, marker="o", color="#E45756")
    plt.xlabel("k (clusters)")
    plt.ylabel("KMeans inertia")
    plt.title("Elbow Method")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(path, show=show)


def visualize_clusters_2d(data2d: np.ndarray, labels: np.ndarray, path: Optional[str] = None, *, title: Optional[str] = None, show: bool = False):
    scatter_2d(data2d, labels=labels, path=path, title=title, cmap='tab10', s=6, alpha=0.8, show=show)


def visualize_clusters_3d(data3d: np.ndarray, labels: np.ndarray, path: Optional[str] = None, *, title: Optional[str] = None, show: bool = False):
    scatter_3d(data3d, labels=labels, path=path, title=title, cmap='tab20', s=1, alpha=0.6, show=show)


def visualize_3d_points(points3d: np.ndarray,
                        path: Optional[str] = None,
                        *,
                        title: Optional[str] = None,
                        color: str = 'blue',
                        s: float = 0.1,
                        alpha: float = 1.0,
                        show: bool = False):
    """
    Simple 3D scatter without labels, mirroring notebook's Visualize3D.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], s=s, color=color, alpha=alpha)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    if title:
        ax.set_title(title)
    plt.tight_layout()
    _save_or_show(path, show=show)


def compare_3d_true_vs_pred(points3d: np.ndarray,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            path: Optional[str] = None,
                            *,
                            title: Optional[str] = None,
                            sil_true: Optional[float] = None,
                            sil_pred: Optional[float] = None,
                            cmap_true: str = 'coolwarm',
                            cmap_pred: str = 'tab20',
                            s: float = 1,
                            alpha: float = 0.6,
                            show: bool = False):
    """
    Side-by-side 3D scatters colored by true labels and predicted clusters,
    similar to the notebook's dual 3D visualization.
    """
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], c=y_true, cmap=cmap_true, s=s, alpha=alpha)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    t1 = 'PCA - True Labels'
    if sil_true is not None:
        t1 += f' (Silhouette: {sil_true:.4f})'
    ax1.set_title(t1)
    plt.colorbar(sc1, ax=ax1, label='Label')

    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], c=y_pred, cmap=cmap_pred, s=s, alpha=alpha)
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_zlabel('PC3')
    t2 = 'PCA - KMeans Clusters'
    if sil_pred is not None:
        t2 += f' (Silhouette: {sil_pred:.4f})'
    ax2.set_title(t2)
    plt.colorbar(sc2, ax=ax2, label='Cluster')

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    _save_or_show(path, show=show)
