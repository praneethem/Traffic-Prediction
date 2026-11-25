"""Checks behind scripts/run_pca_study.py.

PCA on standardised data is an orthogonal rotation. Ridge is rotation invariant, so Ridge on all components must give
the same predictions as Ridge on the original standardised features. That is why "PCA + Ridge" with k = 31 equals
Ridge in the benchmark, and why the study can use the k = 31 row as a sanity check."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def _data(n=2000, p=12, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 4))
    X = np.column_stack([base @ rng.normal(size=(4, p - 2)), rng.normal(size=(n, 2))])   # correlated columns, like lags
    y = X[:, 0] - 0.5 * X[:, 3] + 0.1 * rng.normal(size=n)
    return X, y


def test_full_pca_ridge_equals_ridge():
    X, y = _data()
    Z = StandardScaler().fit_transform(X)
    T = PCA(n_components=Z.shape[1], svd_solver="full", random_state=0).fit_transform(Z)
    a = Ridge(alpha=0.5).fit(Z, y).predict(Z)
    b = Ridge(alpha=0.5).fit(T, y).predict(T)
    assert np.allclose(a, b, atol=1e-8)


def test_fewer_components_lose_fit_and_variance_adds_up():
    X, y = _data()
    Z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=Z.shape[1], svd_solver="full", random_state=0).fit(Z)
    assert np.isclose(pca.explained_variance_ratio_.sum(), 1.0)
    T = pca.transform(Z)
    err = {k: np.abs(Ridge(alpha=0.5).fit(T[:, :k], y).predict(T[:, :k]) - y).mean() for k in (2, 6, 12)}
    assert err[2] > err[6] >= err[12] - 1e-9
