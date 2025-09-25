# model/infer/runner.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.utils.validation import check_is_fitted

from model.infer.prep import expand_like_training, time_split_df
from model.infer.registry import IMPORTANCE_REGISTRY
from model.repository import ArtifactRepository
from utils.logger import get_logger

logger = get_logger("model.infer.runner")


def _ensure_test_matrices_for_methods(
    *, fe, raw_path: str, cutoff: str
) -> tuple[sp.csr_matrix, np.ndarray | None, pd.DataFrame]:
    """Prepare TEST matrices for permutation/SHAP (FE is already fitted in artifact)."""
    logger.info("Loading raw data: %s", raw_path)
    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)
    df_raw = pd.DataFrame(raw)

    date_col = getattr(fe, "date_col", "date")
    df_wide = expand_like_training(df_raw, fe)
    _, df_test = time_split_df(df_wide, cutoff=cutoff, date_col=date_col)
    if df_test.empty:
        raise ValueError("No TEST rows after cutoff; cannot compute importances.")

    X_test_sparse = fe.transform(df_test)

    target_col = getattr(fe, "target_user_col", "user_id")
    target_val = getattr(fe, "target_user_value", 0)
    y_test = (
        (df_test[target_col] == target_val).astype(int).to_numpy()
        if target_col in df_test.columns
        else None
    )
    return X_test_sparse, y_test, df_test


def run_feature_importance(
    *,
    model_path: str,
    out_csv: str,
    topk: int | None,
    method: str,  # "built_in" | "permutation" | "shap"
    raw_path: str | None,
    cutoff: str | None,
    n_repeats: int = 5,
    seed: int = 42,
    scoring: str = "roc_auc",
    perm_rows: int = 4000,
    allow_dense_full: bool = False,
    plot: bool = False,
    plot_path: str | None = None,
) -> Path:
    out_csv_p = Path(out_csv)
    out_csv_p.parent.mkdir(parents=True, exist_ok=True)

    repo = ArtifactRepository()
    art = repo.load(model_path)
    pipeline = art.pipeline
    fe = art.fe
    model = art.model
    check_is_fitted(model)

    if not hasattr(fe, "feature_names_"):
        raise ValueError("The saved feature engineer has no 'feature_names_'.")

    fn = IMPORTANCE_REGISTRY.get(method)
    if fn is None:
        raise ValueError(f"Unknown method '{method}'. Available: {sorted(IMPORTANCE_REGISTRY)}")

    if method == "built_in":
        df = fn(model=model, feature_names=list(fe.feature_names_), topk=topk)
    else:
        if not raw_path or not cutoff:
            raise ValueError(f"--method {method} requires --raw and --cutoff")
        X_test_sparse, y_test, _ = _ensure_test_matrices_for_methods(
            fe=fe, raw_path=raw_path, cutoff=cutoff
        )
        df = fn(
            model=model,
            X_test_sparse=X_test_sparse,
            y_test=y_test,
            feature_names=list(fe.feature_names_),
            prefilter_topk=topk,  # use topk as budget for permutation prefilter
            n_repeats=n_repeats,
            scoring=scoring,
            random_state=seed,
            perm_rows=perm_rows,
            allow_dense_full=allow_dense_full,
            sample_rows=perm_rows or 4000,  # for SHAP
            topk=topk,
        )

    df.to_csv(out_csv_p, index=False)
    logger.info("Wrote feature importances to %s", out_csv_p.resolve())

    if plot:
        import matplotlib.pyplot as plt

        p = Path(plot_path) if plot_path else out_csv_p.with_suffix(".png")
        d = df.head(topk) if topk else df
        d = d.iloc[::-1]
        plt.figure(figsize=(10, max(4, 0.35 * len(d))))
        plt.barh(d["feature"], d["importance"])
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title(f"{type(model).__name__} — {method} importance (top {topk or len(df)})")
        plt.tight_layout()
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=150)
        plt.close()
        logger.info("Wrote importance plot to %s", p.resolve())

    return out_csv_p
