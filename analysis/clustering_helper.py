import numpy as np
from typing import Tuple, List, Optional

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except Exception:
    KMeans = None
    silhouette_score = None


def kmeans_cluster(
    X: np.ndarray,
    n_clusters: int,
    *,
    random_state: Optional[int] = 42,
    n_init: int = 10,
    max_iter: int = 300,
) -> Tuple[np.ndarray, float, np.ndarray, Optional[float]]:
    """Run KMeans clustering.

    Returns (labels, inertia, centers, silhouette) where silhouette may be None
    if `sklearn.metrics.silhouette_score` is unavailable or degenerate labels.
    """
    if KMeans is None:
        raise ImportError(
            "scikit-learn is required for KMeans. Install with `pip install scikit-learn`."
        )
    if X.dtype != np.float32 and X.dtype != np.float64:
        X = X.astype(np.float32)
    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )
    labels = km.fit_predict(X)
    inertia = float(km.inertia_)
    centers = km.cluster_centers_
    sil = None
    if silhouette_score is not None and len(np.unique(labels)) > 1:
        try:
            sil = float(silhouette_score(X, labels, metric="euclidean"))
        except Exception:
            sil = None
    return labels, inertia, centers, sil


def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score for labels on X. Returns NaN if not possible."""
    if silhouette_score is None:
        raise ImportError(
            "scikit-learn is required for silhouette_score. Install with `pip install scikit-learn`."
        )
    if len(np.unique(labels)) < 2:
        return float("nan")
    if X.dtype != np.float32 and X.dtype != np.float64:
        X = X.astype(np.float32)
    return float(silhouette_score(X, labels, metric="euclidean"))


def elbow_inertia(X: np.ndarray, k_values: List[int], *, random_state: Optional[int] = 42) -> List[float]:
    """Compute KMeans inertia over a list of k values to plot an elbow curve."""
    inertias: List[float] = []
    for k in k_values:
        labels, inertia, _, _ = kmeans_cluster(X, n_clusters=k, random_state=random_state)
        inertias.append(inertia)
    return inertias
