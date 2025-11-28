import numpy as np
from typing import Tuple, Optional
import logging

try:
    from sklearn.decomposition import PCA
except Exception as e:
    PCA = None


logger = logging.getLogger(__name__)


def compute_pca(
    X: np.ndarray,
    n_components: int = 16,
    *,
    whiten: bool = False,
    random_state: Optional[int] = 42,
    svd_solver: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, object]:
    """Compute PCA projection for dataset X.

    Returns (projected, explained_variance_ratio, pca_model).
    """
    logger.debug("PCA compute_pca: dtype=%s shape=%s n_components=%d whiten=%s svd_solver=%s random_state=%s",
                 getattr(X, 'dtype', None), getattr(X, 'shape', None), n_components, whiten, svd_solver, str(random_state))
    if PCA is None:
        raise ImportError(
            "scikit-learn is required for PCA. Install with `pip install scikit-learn`."
        )
    if X.dtype != np.float32 and X.dtype != np.float64:
        logger.debug("PCA compute_pca: casting X to float32 from %s", X.dtype)
        X = X.astype(np.float32)
    pca = PCA(
        n_components=n_components,
        whiten=whiten,
        random_state=random_state,
        svd_solver=svd_solver,
    )
    logger.debug("PCA fit_transform starting...")
    projected = pca.fit_transform(X)
    logger.debug("PCA fit_transform done: projected.shape=%s, EVR(first 5)=%s",
                 getattr(projected, 'shape', None), pca.explained_variance_ratio_[:5])
    return projected, pca.explained_variance_ratio_, pca


def pca_eigen_info(
    X: np.ndarray,
    n_components: Optional[int] = None,
    *,
    random_state: Optional[int] = 42,
    svd_solver: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (eigenvalue_ratios, eigenvectors) for dataset X.

    If n_components is None, will use full rank (min(n_samples, n_features)).
    """
    logger.debug("PCA pca_eigen_info: dtype=%s shape=%s n_components=%s svd_solver=%s random_state=%s",
                 getattr(X, 'dtype', None), getattr(X, 'shape', None), str(n_components), svd_solver, str(random_state))
    if PCA is None:
        raise ImportError(
            "scikit-learn is required for PCA. Install with `pip install scikit-learn`."
        )
    if X.dtype != np.float32 and X.dtype != np.float64:
        logger.debug("PCA pca_eigen_info: casting X to float32 from %s", X.dtype)
        X = X.astype(np.float32)
    pca = PCA(
        n_components=n_components,
        random_state=random_state,
        svd_solver=svd_solver,
    )
    logger.debug("PCA fit starting...")
    pca.fit(X)
    logger.debug("PCA fit done: components.shape=%s, EVR(first 5)=%s",
                 getattr(pca.components_, 'shape', None), pca.explained_variance_ratio_[:5])
    # components_: shape (n_components, n_features)
    return pca.explained_variance_ratio_, pca.components_
