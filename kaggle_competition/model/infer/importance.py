from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from model.infer.registry import register_importance
from utils.logger import get_logger

logger = get_logger("model.infer.importance")


# ----------------------------- small helpers -----------------------------


def _as_dense_slice(
    X_sparse: sp.spmatrix,
    y: np.ndarray | None,
    perm_rows: int,
    allow_dense_full: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Safely convert a sparse TEST slice to dense for permutation importance.

    - If perm_rows > 0 and TEST is bigger, keep only the most recent perm_rows rows.
    - If perm_rows == 0 and allow_dense_full is False, raise to protect memory.
    """
    n = X_sparse.shape[0]
    if perm_rows and n > perm_rows:
        idx = np.arange(n - perm_rows, n)
        X_out = X_sparse[idx, :].toarray()
        y_out = y[idx] if y is not None else None
        logger.info(
            "Permutation: downsampled TEST from %d to %d rows for dense conversion.", n, perm_rows
        )
        return X_out, y_out

    if not allow_dense_full and not perm_rows and n > 8000:
        raise ValueError(
            f"Converting full TEST ({n} rows) to dense could be large. "
            f"Either set --perm_rows (e.g., 4000) or pass --allow_dense_full."
        )

    logger.info("Permutation: using FULL TEST (%d rows) for dense conversion.", n)
    return X_sparse.toarray(), y


def _built_in_values(model: Any) -> np.ndarray:
    """Fetch built-in importances (LGBM: feature_importances_, Linear: |coef|)."""
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_)
        if imp.ndim == 2:
            imp = imp.mean(axis=0)
        return imp.astype(float)
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 2 and coef.shape[0] == 1:
            coef = coef[0]
        if coef.ndim == 2:
            coef = np.linalg.norm(coef, axis=0)
        return np.abs(coef).astype(float)
    raise ValueError("This model type does not expose built-in feature importances.")


def _as_df(
    feature_names: list[str],
    values: np.ndarray,
    *,
    method: str,
    model_name: str,
) -> pd.DataFrame:
    """Standardized importance DataFrame."""
    values = np.asarray(values, dtype=float).ravel()
    if len(feature_names) != len(values):
        raise ValueError(
            f"feature_names length ({len(feature_names)}) != values length ({len(values)})"
        )
    total = float(values.sum()) if float(values.sum()) != 0.0 else 1.0
    df = pd.DataFrame({"feature": feature_names, "importance": values})
    df["pct_of_total"] = (df["importance"] / total) * 100.0
    df["method"] = method
    df["model_type"] = model_name
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def _prefilter_by_builtin(model: Any, feature_names: list[str], topk: int) -> np.ndarray:
    """Return indices of top-k features by built-in importances."""
    values = _built_in_values(model)
    if len(values) != len(feature_names):
        raise ValueError("Length mismatch between built-in importance and feature_names.")
    order = np.argsort(-values)
    keep = order[: min(topk, len(order))]
    logger.info(
        "Prefilter by built-in importance: keeping top-%d of %d features.", len(keep), len(order)
    )
    return keep


def _permutation_subset(
    model: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: np.ndarray,
    *,
    n_repeats: int,
    scoring: str = "roc_auc",
    random_state: int = 42,
) -> np.ndarray:
    """
    Fast(er) permutation importance over a subset of columns.

    - Computes baseline score once.
    - For each feature in feature_indices:
       repeat n_repeats times: shuffle that column, score, and average drop.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    if scoring not in {"roc_auc", "average_precision"}:
        raise ValueError("Supported scoring: 'roc_auc' or 'average_precision'.")

    rng = np.random.RandomState(random_state)

    # Baseline proba/score
    if hasattr(model, "predict_proba"):
        base_proba = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        s = model.decision_function(X)
        base_proba = (s - s.min()) / (s.max() - s.min() + 1e-12)
    else:
        base_proba = model.predict(X).astype(float)

    def _score_from_proba(p: np.ndarray) -> float:
        if scoring == "roc_auc":
            return float(roc_auc_score(y, p))
        return float(average_precision_score(y, p))

    base_score = _score_from_proba(base_proba)
    logger.info("Permutation baseline %s = %.6f", scoring, base_score)

    importances = np.zeros(len(feature_indices), dtype=float)

    # Work on a copy to avoid reallocating big arrays per repeat
    X_work = X.copy()

    for k, j in enumerate(feature_indices):
        drops: list[float] = []
        original_col = X[:, j].copy()
        for _ in range(n_repeats):
            rng.shuffle(X_work[:, j])
            # score
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X_work)[:, 1]
            elif hasattr(model, "decision_function"):
                s = model.decision_function(X_work)
                p = (s - s.min()) / (s.max() - s.min() + 1e-12)
            else:
                p = model.predict(X_work).astype(float)
            drops.append(base_score - _score_from_proba(p))
        importances[k] = float(np.mean(drops))
        # restore column
        X_work[:, j] = original_col

        if (k + 1) % 50 == 0:
            logger.info("Permutation: processed %d/%d features...", k + 1, len(feature_indices))

    return importances


# ----------------------------- strategies -----------------------------


@register_importance("built_in")
def run_built_in(
    *,
    model: Any,
    feature_names: list[str],
    topk: int | None = None,
    **_: Any,
) -> pd.DataFrame:
    """Built-in importances (tree feature_importances_, linear |coef|)."""
    check_is_fitted(model)
    values = _built_in_values(model)
    df = _as_df(feature_names, values, method="built_in", model_name=type(model).__name__)
    return df.head(int(topk)).copy() if topk else df


@register_importance("permutation")
def run_permutation(
    *,
    model: Any,
    X_test_sparse: sp.spmatrix,
    y_test: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42,
    perm_rows: int = 4000,
    allow_dense_full: bool = False,
    prefilter_topk: int = 200,
    topk: int | None = None,
    **_: Any,
) -> pd.DataFrame:
    """
    Permutation importance on TEST (dense slice for speed).
    Optionally prefilter to top-k by built-in importances to speed up.
    """
    check_is_fitted(model)

    X_dense, y_dense_opt = _as_dense_slice(X_test_sparse, y_test, perm_rows, allow_dense_full)
    if y_dense_opt is None:
        raise ValueError("Permutation importance requires labels (y_test).")
    y_dense = y_dense_opt

    if prefilter_topk and prefilter_topk > 0:
        try:
            idx_keep = _prefilter_by_builtin(model, feature_names, prefilter_topk)
        except Exception as e:
            logger.warning(
                "Prefilter failed (%s). Falling back to first topk=%d columns.",
                e,
                prefilter_topk,
            )
            idx_keep = np.arange(min(prefilter_topk, X_dense.shape[1]))
    else:
        idx_keep = np.arange(X_dense.shape[1])

    vals_subset = _permutation_subset(
        model=model,
        X=X_dense,
        y=y_dense,
        feature_indices=idx_keep,
        n_repeats=n_repeats,
        scoring=scoring,
        random_state=random_state,
    )

    # fill full vector with zeros for features not permuted (stable shape)
    vals_full = np.zeros(X_dense.shape[1], dtype=float)
    vals_full[idx_keep] = vals_subset

    df = _as_df(feature_names, vals_full, method=f"perm_{scoring}", model_name=type(model).__name__)
    return df.head(int(topk)).copy() if topk else df


@register_importance("shap")
def run_shap_lgbm(
    *,
    model: Any,
    X_test_sparse: sp.spmatrix,
    feature_names: list[str],
    topk: int | None = None,
    sample_rows: int = 4000,
    **_: Any,
) -> pd.DataFrame:
    """
    SHAP values for LightGBM-like models: fast, sparse-friendly.
    Returns DataFrame with mean |SHAP| per feature.
    """
    try:
        import shap  # type: ignore
    except Exception as exc:  # pragma: no cover - import-time dependency
        raise ImportError("Please `pip install shap` to use --method shap") from exc

    # sample rows (keep most recent rows)
    n = X_test_sparse.shape[0]
    if n > sample_rows:
        idx = np.arange(n - sample_rows, n)
        X_use = X_test_sparse[idx, :]
    else:
        X_use = X_test_sparse

    logger.info("Computing SHAP on %d rows...", X_use.shape[0])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_use, check_additivity=False)

    # LightGBM binary often returns [neg, pos]; prefer positive class if list
    if isinstance(shap_values, list):
        shap_arr = np.asarray(shap_values[1] if len(shap_values) >= 2 else shap_values[0])
    else:
        shap_arr = np.asarray(shap_values)

    if sp.issparse(shap_arr):  # type: ignore[arg-type]
        shap_arr = shap_arr.toarray()  # type: ignore[assignment]

    vals = np.abs(np.asarray(shap_arr)).mean(axis=0).ravel()

    df = _as_df(feature_names, vals, method="shap_mean_abs", model_name=type(model).__name__)
    return df.head(int(topk)).copy() if topk else df
