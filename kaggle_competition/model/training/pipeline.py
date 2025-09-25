# model/training/pipeline.py
from __future__ import annotations

from sklearn.pipeline import Pipeline

from model.algorithms import make_lgbm, make_logreg

from .registry import register_model


def build_pipeline(fe, base_estimator) -> Pipeline:
    """FE + estimator in one Pipeline (prevents leakage; persists cleanly)."""
    return Pipeline([("fe", fe), ("clf", base_estimator)])


def _p(step: str = "clf", **params):
    """Prefix params for sklearn Pipeline."""
    return {f"{step}__{k}": v for k, v in params.items()}


@register_model("logreg")
def space_logreg(fe, scale_pos_weight: float, seed: int) -> dict:
    from scipy.stats import loguniform

    est = make_logreg(random_state=seed)
    pipe = build_pipeline(fe, est)
    return {
        "estimator": pipe,
        "params": _p(C=loguniform(1e-3, 1e3)),
    }


@register_model("lgbm")
def space_lgbm(fe, scale_pos_weight: float, seed: int) -> dict:
    from scipy.stats import loguniform, randint, uniform

    est = make_lgbm(scale_pos_weight=scale_pos_weight, random_state=seed, n_jobs=1)
    pipe = build_pipeline(fe, est)
    return {
        "estimator": pipe,
        "params": _p(
            n_estimators=randint(300, 1500),
            learning_rate=loguniform(1e-3, 3e-1),
            num_leaves=randint(31, 255),
            min_child_samples=randint(10, 200),
            subsample=uniform(0.6, 0.4),
            colsample_bytree=uniform(0.6, 0.4),
            reg_lambda=loguniform(1e-3, 10),
        ),
    }
