# model/algorithms.py
from __future__ import annotations

from typing import Any

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression


def make_logreg(**kwargs: Any) -> LogisticRegression:
    """
    Logistic Regression tuned for sparse features and imbalance.
    """
    params = dict(
        penalty="l2",
        C=1.0,
        solver="liblinear",  # robust & works well on sparse binary problems
        class_weight="balanced",
        max_iter=1000,
        n_jobs=1,  # liblinear ignores n_jobs>1
    )
    params.update(kwargs)
    return LogisticRegression(**params)


def make_lgbm(*, scale_pos_weight: float | None = None, **kwargs: Any):
    """
    LightGBM binary classifier. Pass scale_pos_weight for imbalance.
    """
    params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary",
        n_jobs=-1,
    )
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = float(scale_pos_weight)
    else:
        params["class_weight"] = "balanced"
    params.update(kwargs)
    return LGBMClassifier(**params)
