# model/infer/prep.py
from __future__ import annotations

import numpy as np
import pandas as pd

from data.prepare import normalize_location, sites_to_wide


def infer_k_from_fe(fe) -> int:
    if getattr(fe, "site_cols_", None):
        return max(int(c.split("_")[1]) for c in fe.site_cols_)
    return 15


def expand_like_training(df_raw: pd.DataFrame, fe) -> pd.DataFrame:
    """Expand 'sites' -> site_i/length_i, normalize location,
    ensure cat/date/time columns exist."""
    df = df_raw.copy()
    k = infer_k_from_fe(fe)

    if "sites" in df.columns:
        wide = df["sites"].apply(lambda x: sites_to_wide(x, k))
        df = pd.concat([df.drop(columns=["sites"]), wide], axis=1)
    else:
        for i in range(1, k + 1):
            if f"site_{i}" not in df:
                df[f"site_{i}"] = np.nan
            if f"length_{i}" not in df:
                df[f"length_{i}"] = np.nan

    df = normalize_location(df)

    for c in getattr(fe, "cat_cols", []):
        if c not in df.columns:
            df[c] = np.nan

    for c in [getattr(fe, "date_col", "date"), getattr(fe, "time_col", "time")]:
        if c not in df.columns:
            df[c] = pd.NaT

    return df


def time_split_df(
    df_wide: pd.DataFrame, cutoff: str, date_col: str = "date"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dt = pd.to_datetime(df_wide[date_col], errors="coerce")
    mask = dt < pd.to_datetime(cutoff)
    return df_wide.loc[mask].reset_index(drop=True), df_wide.loc[~mask].reset_index(drop=True)


def labels_from_df(df: pd.DataFrame, fe) -> np.ndarray | None:
    target_col = getattr(fe, "target_user_col", "user_id")
    target_val = getattr(fe, "target_user_value", 0)
    if target_col not in df.columns:
        return None
    return (df[target_col] == target_val).astype(int).to_numpy()
