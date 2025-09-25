from typing import Any

from sklearn.base import ClassifierMixin


def fit_model(estimator: ClassifierMixin, X, y, **fit_kwargs: Any) -> ClassifierMixin:
    # LightGBM + sklearn both accept sparse CSR; early stopping optional via **fit_kwargs
    return estimator.fit(X, y, **fit_kwargs)
