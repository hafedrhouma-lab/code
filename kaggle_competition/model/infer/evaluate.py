# model/infer/evaluate.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model.cv import ForwardTimeSplit
from model.evaluate import (
    _proba,
    cv_thresholds_forward,
    evaluate_binary,
    subset_rows,
    threshold_for_recall,
)
from model.infer.prep import expand_like_training, labels_from_df, time_split_df
from model.repository import ArtifactRepository
from utils.logger import get_logger

logger = get_logger("infer.evaluate")


def _parse_thresholds_arg(val: str) -> list[float]:
    parts = [p.strip() for p in val.split(",") if p.strip() != ""]
    return [float(p) for p in parts]


def recompute_threshold_pipeline(
    pipeline,
    X_train_raw,
    y_train,
    train_dates: np.ndarray,
    n_splits: int,
    recall_target: float,
    agg: str,
    use_max_precision: bool,
) -> float:
    splitter = ForwardTimeSplit(dates=train_dates, n_splits=n_splits, min_train_size=1)
    splits = list(splitter.split())
    thrs = cv_thresholds_forward(
        base_estimator=pipeline,
        X=X_train_raw,
        y=y_train,
        splits=splits,
        recall_target=recall_target,
        use_max_precision=use_max_precision,
    )
    if len(thrs) == 0:
        order = np.argsort(train_dates)
        cut = int(len(order) * 0.9)
        val_idx = order[cut:]
        X_val = subset_rows(X_train_raw, val_idx)  # <-- safe row subset
        return threshold_for_recall(y_train[val_idx], _proba(pipeline, X_val), target=recall_target)
    q = {"median": 0.5, "p25": 0.25, "p10": 0.10}[agg]
    return float(np.quantile(thrs, q))


def run_evaluation(
    *,
    model_path: str,
    raw_path: str,
    cutoff: str,
    out_dir: str,
    manual_threshold: float | None,
    manual_thresholds: str | None,
    recompute_threshold: bool,
    n_splits: int,
    recall_target: float,
    thr_agg: str,
    thr_use_max_precision: bool,
) -> Path:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    logger.info("Loading artifact: %s", model_path)
    repo = ArtifactRepository()
    art = repo.load(model_path)
    pipeline = art.pipeline
    fe = art.fe
    stored_threshold = 0.5 if art.threshold is None else float(art.threshold)

    logger.info("Loading raw data: %s", raw_path)
    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)
    df_raw = pd.DataFrame(raw)

    # Expand like training and split
    df_wide = expand_like_training(df_raw, fe)
    date_col = getattr(fe, "date_col", "date")
    df_train, df_test = time_split_df(df_wide, cutoff=cutoff, date_col=date_col)
    logger.info("Rows -> TRAIN=%d, TEST=%d", len(df_train), len(df_test))

    # Labels (if present)
    y_train = labels_from_df(df_train, fe)
    y_test = labels_from_df(df_test, fe)

    # Decide thresholds to evaluate
    manual_mode = False
    thresholds_to_eval: list[float] = []

    if manual_thresholds:
        thresholds_to_eval = _parse_thresholds_arg(manual_thresholds)
        manual_mode = True
        logger.info("Manual thresholds: %s", thresholds_to_eval)
    elif manual_threshold is not None:
        thresholds_to_eval = [float(manual_threshold)]
        manual_mode = True
        logger.info("Manual threshold: %.6f", thresholds_to_eval[0])

    if not manual_mode:
        threshold = stored_threshold
        if recompute_threshold and y_train is not None and len(df_train) > 0:
            logger.info("Recomputing robust threshold via forward-chaining CV on raw TRAIN ...")
            train_dates = pd.to_datetime(df_train[date_col], errors="coerce").to_numpy()
            threshold = recompute_threshold_pipeline(
                pipeline=pipeline,
                X_train_raw=df_train,  # raw DF; Pipeline refits FE per fold
                y_train=y_train,
                train_dates=train_dates,
                n_splits=n_splits,
                recall_target=recall_target,
                agg=thr_agg,
                use_max_precision=thr_use_max_precision,
            )
            logger.info("Recomputed threshold: %.6f", threshold)
        else:
            logger.info("Using stored threshold: %.6f", threshold)
        thresholds_to_eval = [threshold]

    # Evaluate (use Pipeline directly on raw)
    results: dict[str, dict[str, Any]] = {"train": {}, "test": {}}
    if y_train is not None and len(df_train) > 0:
        for thr in thresholds_to_eval:
            results["train"][f"{thr:.6f}"] = evaluate_binary(
                pipeline, df_train, y_train, threshold=thr
            )
    else:
        logger.warning("No labels found in TRAIN; skipping TRAIN metrics.")
    if y_test is not None and len(df_test) > 0:
        for thr in thresholds_to_eval:
            results["test"][f"{thr:.6f}"] = evaluate_binary(
                pipeline, df_test, y_test, threshold=thr
            )
    else:
        logger.warning("No labels found in TEST; skipping TEST metrics.")

    out_path = out_dir_p / f"evaluate_{Path(model_path).stem}.json"
    meta = {
        "cutoff": cutoff,
        "mode": "manual" if manual_mode else "auto",
        "thresholds_evaluated": thresholds_to_eval,
        "thr_aggregation": thr_agg
        if (not manual_mode and recompute_threshold)
        else ("manual" if manual_mode else "stored"),
        "recall_target": recall_target if (not manual_mode and recompute_threshold) else None,
        "metrics": results,
        "model_path": str(Path(model_path).resolve()),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved evaluation summary to %s", out_path.resolve())
    return out_path
