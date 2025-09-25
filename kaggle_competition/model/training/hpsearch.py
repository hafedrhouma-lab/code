# model/training/hpsearch.py
from __future__ import annotations

import importlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from data.io import load_raw_df
from data.prepare import DatasetBundle, PrepareConfig, prepare_dataset_raw
from model.cv import ForwardTimeSplit
from model.evaluate import (
    _proba,
    cv_thresholds_forward,
    evaluate_binary,
    make_precision_at_recall_target_scorer,
    subset_rows,
    threshold_for_recall,
)
from model.repository import ArtifactRepository
from model.training.registry import MODEL_REGISTRY
from utils.logger import get_logger


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def prepare_data(
    raw_path: str, cutoff: str, max_tfidf: int, top_sig: int, logger
) -> tuple[DatasetBundle, np.ndarray]:
    df_raw = load_raw_df(raw_path)
    prep_cfg = PrepareConfig(
        cutoff_date=cutoff,
        max_tfidf_sites=max_tfidf,
        top_user_signature_sites=top_sig,
    )
    bundle = prepare_dataset_raw(df_raw, prep_cfg)
    logger.info(
        "Prepared dataset (raw) with cutoff=%s: TRAIN=%d, TEST=%d",
        cutoff,
        bundle.X_train_raw.shape[0],
        bundle.X_test_raw.shape[0],
    )
    train_dates = pd.to_datetime(bundle.df_train[prep_cfg.date_col], errors="coerce").to_numpy()
    return bundle, train_dates


def build_spaces(fe, y_train: np.ndarray, seed: int, logger) -> dict[str, dict]:
    """
    Build model family search spaces from the registry.
    Ensures registration modules are imported before reading the registry.
    """
    # Force-load the module that registers model families (logreg, lgbm, etc.)
    importlib.import_module("model.training.pipeline")

    if not MODEL_REGISTRY:
        raise RuntimeError(
            "No model families registered. Ensure 'model/training/pipeline.py' defines "
            "@register_model(...) functions and that the module is importable."
        )

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    spw = max(neg / max(pos, 1), 1.0)

    spaces = {name: fn(fe, spw, seed) for name, fn in MODEL_REGISTRY.items()}
    logger.info("Registered model families: %s", ", ".join(sorted(MODEL_REGISTRY)))
    return spaces


def run_family_search(
    name: str,
    estimator_spec: dict,
    X_train_raw,
    y_train,
    splitter,
    n_iter: int,
    n_jobs: int,
    seed: int,
    recall_target: float,
    out_dir: Path,
    logger,
) -> tuple[str, RandomizedSearchCV]:
    logger.info("HP search: %s (n_iter=%d, folds=%d)", name, n_iter, splitter.n_splits)
    prec_scorer = make_precision_at_recall_target_scorer(recall_target=recall_target)
    search = RandomizedSearchCV(
        estimator=estimator_spec["estimator"],
        param_distributions=estimator_spec["params"],
        n_iter=n_iter,
        scoring={"ap": "average_precision", "prec_at_rec": prec_scorer},
        refit="prec_at_rec",
        n_jobs=n_jobs,
        pre_dispatch="2*n_jobs",
        cv=splitter,
        random_state=seed,
        verbose=1,
    )
    search.fit(X_train_raw, y_train)

    idx = search.best_index_
    mean_prec_at_rec = float(search.cv_results_["mean_test_prec_at_rec"][idx])
    mean_ap = float(search.cv_results_["mean_test_ap"][idx])
    logger.info(
        "%s best prec@rec=%.6f (cv_ap=%.6f) params=%s",
        name,
        mean_prec_at_rec,
        mean_ap,
        json.dumps(search.best_params_),
    )

    family_dir = out_dir / name
    _ensure_dir(family_dir)
    with open(family_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_prec_at_rec": mean_prec_at_rec,
                "best_cv_ap": mean_ap,
                "best_params": search.best_params_,
            },
            f,
            indent=2,
        )

    return name, search


def select_best(searches: list[tuple[str, RandomizedSearchCV]]) -> tuple[str, RandomizedSearchCV]:
    if not searches:
        raise RuntimeError("No searches were run. Check the registry and search configuration.")
    return max(
        searches,
        key=lambda x: (
            x[1].cv_results_["mean_test_prec_at_rec"][x[1].best_index_],
            x[1].cv_results_["mean_test_ap"][x[1].best_index_],
        ),
    )


def compute_threshold(
    estimator,
    X_train_raw,
    y_train,
    splitter,
    train_dates: np.ndarray,
    recall_target: float,
    thr_agg: str,
    use_max_precision: bool,
    logger,
) -> float:
    splits = list(splitter.split())
    thrs = cv_thresholds_forward(
        base_estimator=estimator,
        X=X_train_raw,
        y=y_train,
        splits=splits,
        recall_target=recall_target,
        use_max_precision=use_max_precision,
    )
    if len(thrs) == 0:
        logger.warning(
            "No thresholds from CV; " "falling back to recall-based threshold on latest TRAIN tail."
        )
        order = np.argsort(train_dates)
        cut = int(len(order) * 0.9)
        cut = min(max(cut, 0), len(order) - 1)
        val_idx = order[cut:]
        X_val = subset_rows(X_train_raw, val_idx)  # <-- safe row subset here
        return threshold_for_recall(
            y_train[val_idx],
            _proba(estimator, X_val),
            target=recall_target,
        )
    q = 0.50 if thr_agg == "median" else (0.25 if thr_agg == "p25" else 0.10)
    thr = float(np.quantile(thrs, q))
    logger.info("Per-fold thresholds: %s", ", ".join(f"{t:.6f}" for t in thrs))
    logger.info("Aggregated threshold (%s): %.6f", thr_agg, thr)
    return thr


def evaluate_and_save(
    estimator,
    bundle: DatasetBundle,
    thr: float,
    best_name: str,
    cutoff: str,
    recall_target: float,
    thr_agg: str,
    out_dir: Path,
    logger,
) -> None:
    # Evaluate using pipeline on raw TEST
    metrics = evaluate_binary(estimator, bundle.X_test_raw, bundle.y_test, threshold=thr)
    metrics.update(
        {
            "chosen_threshold": float(thr),
            "model_type": best_name,
            "cutoff": cutoff,
            "recall_target": recall_target,
            "thr_aggregation": thr_agg,
        }
    )
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_path = out_dir / f"best_{best_name}_{ts}.joblib"
    repo = ArtifactRepository()
    repo.save(str(model_path), estimator, threshold=thr)
    summary_path = out_dir / f"summary_{ts}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved model: %s", model_path.resolve())
    logger.info("Saved summary: %s", summary_path.resolve())


def run(
    *,
    raw_path: str,
    artifacts: str,
    cutoff: str,
    n_splits: int,
    n_iter: int,
    recall_target: float,
    thr_agg: str,
    seed: int,
    max_tfidf: int,
    top_sig: int,
    n_jobs: int,
    thr_use_max_precision: bool,
):
    logger = get_logger("model.training.hpsearch")
    out_dir = Path(artifacts)
    _ensure_dir(out_dir)
    logger.info("Artifacts directory: %s", out_dir.resolve())

    # 1) Data
    bundle, train_dates = prepare_data(raw_path, cutoff, max_tfidf, top_sig, logger)
    splitter = ForwardTimeSplit(dates=train_dates, n_splits=n_splits, min_train_size=1)

    # 2) Search spaces from registry (ensure module import)
    spaces = build_spaces(bundle.fe, bundle.y_train, seed, logger)

    # 3) Run searches
    results: list[tuple[str, RandomizedSearchCV]] = []
    for name, spec in spaces.items():
        results.append(
            run_family_search(
                name=name,
                estimator_spec=spec,
                X_train_raw=bundle.X_train_raw,
                y_train=bundle.y_train,
                splitter=splitter,
                n_iter=n_iter,
                n_jobs=n_jobs,
                seed=seed,
                recall_target=recall_target,
                out_dir=out_dir,
                logger=logger,
            )
        )

    # 4) Best
    best_name, best_search = select_best(results)
    best_estimator = best_search.best_estimator_
    logger.info("Best-by-CV: %s params=%s", best_name, json.dumps(best_search.best_params_))

    # 5) Threshold
    # 5) Threshold
    thr = compute_threshold(
        estimator=best_estimator,
        X_train_raw=bundle.X_train_raw,
        y_train=bundle.y_train,
        splitter=splitter,
        train_dates=train_dates,
        recall_target=recall_target,
        thr_agg=thr_agg,
        use_max_precision=thr_use_max_precision,
        logger=logger,
    )

    # 6) Evaluate & save
    evaluate_and_save(
        estimator=best_estimator,
        bundle=bundle,
        thr=thr,
        best_name=best_name,
        cutoff=cutoff,
        recall_target=recall_target,
        thr_agg=thr_agg,
        out_dir=out_dir,
        logger=logger,
    )

    logger.info("Done.")
