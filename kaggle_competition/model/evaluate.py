# model/evaluate.py
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    precision_score,
    roc_auc_score,
)

# ---------- Safe utilities ----------


def subset_rows(X, idx: np.ndarray):
    """
    Return a row subset of X for row index positions `idx`, regardless of X type.
    Supports: pandas DataFrame/Series, numpy arrays, scipy sparse matrices.
    """
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    try:
        if sp.issparse(X):
            return X[idx, :]
    except Exception:
        pass
    return X[idx]


def _proba(estimator, X) -> np.ndarray:
    """
    Predict positive-class probabilities (or a score mapped to [0,1]).
    Works for sklearn Pipeline or bare estimators.
    """
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        s = estimator.decision_function(X)
        s = np.asarray(s, dtype=float).ravel()
        # map to [0,1]
        smin, smax = s.min(), s.max()
        return (s - smin) / (smax - smin + 1e-12)
    # fallback: predict labels, cast to float
    p = estimator.predict(X)
    return np.asarray(p, dtype=float).ravel()


# ---------- Metrics & thresholding ----------


def threshold_for_recall(y_true: np.ndarray, proba: np.ndarray, target: float) -> float:
    """
    Smallest threshold that achieves recall >= target.
    If no positives or target cannot be met, returns the minimum viable threshold.
    """
    y = np.asarray(y_true, dtype=int).ravel()
    p = np.asarray(proba, dtype=float).ravel()
    pos = int(y.sum())
    if pos == 0:
        # no positives; any threshold yields recall = 0; return 1.0 to label none
        return 1.0

    order = np.argsort(-p)  # descending by score
    y_sorted = y[order]
    p_sorted = p[order]

    tp_cum = np.cumsum(y_sorted)
    recall = tp_cum / max(pos, 1)

    idx = np.where(recall >= target)[0]
    if len(idx) == 0:
        # can't reach target; set threshold so that everything predicted negative
        return float(p_sorted.max()) + 1e-12

    # first index to reach the target recall => minimal threshold for that recall
    thr = float(p_sorted[idx[0]])
    return thr


def evaluate_binary(estimator, X, y: np.ndarray, threshold: float) -> dict[str, Any]:
    """
    Compute common binary metrics at a fixed decision threshold.
    """
    y = np.asarray(y, dtype=int).ravel()
    proba = _proba(estimator, X)
    pred = (proba >= float(threshold)).astype(int)

    ap = average_precision_score(y, proba) if len(np.unique(y)) > 1 else float("nan")
    roc = roc_auc_score(y, proba) if len(np.unique(y)) > 1 else float("nan")
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    acc = accuracy_score(y, pred)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    return {
        "threshold": float(threshold),
        "ap": float(ap),
        "roc_auc": float(roc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "accuracy": float(acc),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def make_precision_at_recall_target_scorer(recall_target: float):
    """
    Returns a scorer callable with signature (estimator, X, y) that:
      - gets continuous scores via predict_proba/decision_function,
      - finds the smallest threshold achieving recall >= recall_target,
      - returns precision at that threshold.

    This avoids make_scorer() entirely for max compatibility across sklearn versions.
    """

    def _score(estimator, X, y_true):
        y_true = np.asarray(y_true, dtype=int).ravel()
        y_score = _proba(estimator, X)
        thr = threshold_for_recall(y_true, y_score, target=recall_target)
        y_pred = (y_score >= thr).astype(int)
        return precision_score(y_true, y_pred, zero_division=0)

    return _score


# ---------- CV thresholding (forward chaining) ----------


def cv_thresholds_forward(
    *,
    base_estimator,
    X,
    y: np.ndarray,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
    recall_target: float,
    use_max_precision: bool,
) -> list[float]:
    """
    For each (train, val) forward-chaining fold:
      - fit a fresh clone of base_estimator on TRAIN,
      - compute the threshold on VAL to achieve recall_target,
      - if use_max_precision=True, among all thresholds that meet recall_target,
        choose the one that yields MAX precision on VAL (more conservative).
    Returns a list of thresholds (one per fold).
    """
    y = np.asarray(y, dtype=int).ravel()
    thresholds: list[float] = []

    for tr_idx, val_idx in splits:
        est = clone(base_estimator)

        X_tr = subset_rows(X, tr_idx)
        y_tr = y[tr_idx]
        est.fit(X_tr, y_tr)

        X_val = subset_rows(X, val_idx)
        y_val = y[val_idx]
        p_val = _proba(est, X_val)

        if not use_max_precision:
            thr = threshold_for_recall(y_val, p_val, target=recall_target)
            thresholds.append(float(thr))
            continue

        # Conservative mode: among thresholds achieving recall >= target,
        # pick the one with max precision
        # Build threshold candidates from sorted unique scores
        order = np.argsort(-p_val)
        y_sorted = y_val[order]
        p_sorted = p_val[order]

        pos = int(y_sorted.sum())
        if pos == 0:
            thresholds.append(1.0)
            continue

        tp_cum = np.cumsum(y_sorted)
        recall = tp_cum / max(pos, 1)
        idx = np.where(recall >= recall_target)[0]
        if len(idx) == 0:
            thresholds.append(float(p_sorted.max()) + 1e-12)
            continue

        # Among indices meeting recall target, compute precision and choose max
        ks = idx  # candidate cut positions
        preds_pos = ks + 1  # number of predicted positives at each cut
        tp_at_k = tp_cum[ks]
        precision_at_k = tp_at_k / np.maximum(preds_pos, 1)
        best_i = int(ks[np.argmax(precision_at_k)])
        thr = float(p_sorted[best_i])
        thresholds.append(thr)

    return thresholds
