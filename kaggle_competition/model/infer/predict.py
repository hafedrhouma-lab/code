# model/infer/predict.py
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from model.evaluate import _proba
from model.infer.prep import expand_like_training
from model.repository import ArtifactRepository
from utils.logger import get_logger

logger = get_logger("infer.predict")


def run_predict(
    *,
    model_path: str,
    raw_path: str,
    out_csv: str,
    manual_threshold: float | None,
) -> Path:
    out_p = Path(out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading artifact: %s", model_path)
    repo = ArtifactRepository()
    art = repo.load(model_path)
    pipeline = art.pipeline
    fe = art.fe
    thr = (
        float(manual_threshold)
        if manual_threshold is not None
        else (0.5 if art.threshold is None else float(art.threshold))
    )
    logger.info("%s threshold: %.6f", "Manual" if manual_threshold is not None else "Stored", thr)

    logger.info("Loading raw data: %s", raw_path)
    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)
    df_raw = pd.DataFrame(raw)

    logger.info("Preparing inference dataframe (expand sites, normalize, ensure cols)")
    df_inf = expand_like_training(df_raw, fe)

    logger.info("Predicting probabilities and labels (thr=%.6f) via Pipeline", thr)
    probs = _proba(pipeline, df_inf)
    labels = (probs >= thr).astype(int)

    out_df = df_inf.copy()
    out_df["prob"] = probs
    out_df["label"] = labels

    # Order columns nicely
    id_like = [c for c in ["session_id", "id", "user_id"] if c in out_df.columns]
    meta = [
        c
        for c in ["browser", "os", "locale", "gender", "location", "date", "time"]
        if c in out_df.columns
    ]
    cc = [c for c in ["country", "city"] if c in out_df.columns]
    site_cols = sorted(
        [c for c in out_df.columns if c.startswith("site_")],
        key=lambda col: int(col.split("_")[1]) if "_" in col else 10**6,
    )
    length_cols = sorted(
        [c for c in out_df.columns if c.startswith("length_")],
        key=lambda col: int(col.split("_")[1]) if "_" in col else 10**6,
    )
    tail = [
        c
        for c in out_df.columns
        if c not in set(id_like + meta + cc + site_cols + length_cols + ["prob", "label"])
    ]
    ordered_cols = id_like + meta + cc + site_cols + length_cols + tail + ["prob", "label"]
    out_df = out_df.reindex(columns=ordered_cols)

    out_df.to_csv(out_p, index=False)
    logger.info("Wrote predictions to %s", out_p.resolve())
    return out_p
