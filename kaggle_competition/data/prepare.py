# data/prepare.py
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class PrepareConfig:
    max_sites_per_session: int = 15
    cutoff_date: str = "2019-01-01"
    target_user_col: str = "user_id"
    target_user_value: int = 0
    max_tfidf_sites: int = 5000
    top_user_signature_sites: int = 300
    use_cyclic_time: bool = True
    night_hours: Sequence[int] = tuple(range(0, 7))
    cat_cols: tuple[str, ...] = ("locale", "country", "city", "browser", "os", "gender")
    date_col: str = "date"
    time_col: str = "time"


# Raw-only bundle for Pipeline-based training/eval (no leakage)
@dataclass(frozen=True)
class DatasetBundle:
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    df_train: pd.DataFrame
    df_test: pd.DataFrame
    fe: SessionFeatureEngineer  # unfitted; fit happens inside Pipeline
    date_col: str


def sites_to_wide(sites: list[dict] | None, k: int) -> pd.Series:
    lst = (sites or [])[:k]
    out = {}
    for i in range(1, k + 1):
        if i <= len(lst) and isinstance(lst[i - 1], dict):
            d = lst[i - 1]
            out[f"site_{i}"] = d.get("site")
            out[f"length_{i}"] = d.get("length")
        else:
            out[f"site_{i}"] = None
            out[f"length_{i}"] = None
    return pd.Series(out)


def normalize_location(df: pd.DataFrame) -> pd.DataFrame:
    if "location" not in df.columns:
        df = df.copy()
        df["country"] = np.nan
        df["city"] = np.nan
        return df
    parts = df["location"].astype("string").str.split("/", n=1, expand=True)
    df = df.copy()
    df["country"] = parts[0]
    df["city"] = parts[1] if parts.shape[1] > 1 else np.nan
    return df.drop(columns=["location"])


class SessionFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer:
    - __init__ stores params EXACTLY as given (no mutation), so clone() works.
    - Any conversions (list/set) happen at use-sites inside methods.
    """

    def __init__(
        self,
        *,
        target_user_col: str,
        target_user_value: int,
        max_tfidf_sites: int,
        top_user_signature_sites: int,
        use_cyclic_time: bool,
        night_hours: Iterable[int],
        cat_cols: Sequence[str],
        date_col: str,
        time_col: str,
    ):
        # --- store params unchanged (required by sklearn clone) ---
        self.target_user_col = target_user_col
        self.target_user_value = target_user_value
        self.max_tfidf_sites = max_tfidf_sites
        self.top_user_signature_sites = top_user_signature_sites
        self.use_cyclic_time = use_cyclic_time
        self.night_hours = night_hours  # keep as passed (Sequence/Iterable)
        self.cat_cols = cat_cols  # keep as passed (Sequence)
        self.date_col = date_col
        self.time_col = time_col

        # Non-parameter attributes are fine to create here
        self.tfidf = TfidfVectorizer(max_features=max_tfidf_sites, token_pattern=r"[^ \t\n\r\f\v]+")
        # If you're on sklearn <1.2, use 'sparse=True'; >=1.2 supports sparse_output
        try:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

        # Learned attributes
        self.signature_sites_: list[str] | None = None
        self.site_cols_: list[str] | None = None
        self.length_cols_: list[str] | None = None
        self.feature_names_: list[str] | None = None

    # sklearn API
    def fit(self, df: pd.DataFrame, y=None):
        self._discover_columns(df)
        corpus = self._sites_corpus(df)
        self.tfidf.fit(corpus)

        # Ensure required cat cols exist, but don't mutate self.cat_cols
        ohe_cols = self._ensure_cols(df, self.cat_cols)
        self.ohe.fit(df[ohe_cols].fillna("Unknown"))

        self.signature_sites_ = self._build_signature_sites(df)
        return self

    def transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        df = df.reset_index(drop=True).copy()
        X_sites = self.tfidf.transform(self._sites_corpus(df))
        X_sig = self._signature_flags(df)
        len_stats = self._session_length_stats(df)
        X_len = sp.csr_matrix(len_stats.values)
        tfeat = self._time_features(df)
        X_time = sp.csr_matrix(tfeat.values)

        ohe_cols = self._ensure_cols(df, self.cat_cols)
        X_meta = self.ohe.transform(df[ohe_cols].fillna("Unknown"))

        X = sp.hstack([X_sites, X_sig, X_len, X_time, X_meta]).tocsr()
        # Build feature names
        self.feature_names_ = (
            [f"tfidf::{v}" for v in self.tfidf.get_feature_names_out()]
            + [f"user_sig::{s}" for s in (self.signature_sites_ or [])]
            + [f"len::{c}" for c in len_stats.columns.tolist()]
            + [f"time::{c}" for c in tfeat.columns.tolist()]
            + [f"meta::{c}" for c in self.ohe.get_feature_names_out(list(self.cat_cols)).tolist()]
        )
        return X

    # ---- internals ----
    def _discover_columns(self, df: pd.DataFrame) -> None:
        self.site_cols_ = [c for c in df.columns if c.startswith("site_")]
        self.length_cols_ = [c for c in df.columns if c.startswith("length_")]

    @staticmethod
    def _ensure_cols(df: pd.DataFrame, cols: Sequence[str]) -> list[str]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            df[missing] = np.nan
        return list(cols)

    def _row_sites(self, row: pd.Series) -> list[str]:
        vals = [
            row[c]
            for c in (self.site_cols_ or [])
            if c in row and pd.notna(row[c]) and str(row[c]).lower() != "none"
        ]
        return [str(s) for s in vals]

    def _sites_corpus(self, df: pd.DataFrame) -> pd.Series:
        return df.apply(lambda r: " ".join(self._row_sites(r)), axis=1)

    def _build_signature_sites(self, df: pd.DataFrame) -> list[str]:
        if self.target_user_col not in df.columns:
            return []
        mask = (df[self.target_user_col] == self.target_user_value).fillna(False)
        if mask.sum() == 0:
            return []
        chunks = []
        for c in self.site_cols_ or []:
            s = df.loc[mask, c].dropna().astype(str)
            s = s[s.str.lower() != "none"]
            chunks.append(s)
        all_sites = pd.concat(chunks, axis=0) if chunks else pd.Series(dtype="string")
        return all_sites.value_counts().head(self.top_user_signature_sites).index.tolist()

    def _signature_flags(self, df: pd.DataFrame) -> sp.csr_matrix:
        if not self.signature_sites_:
            return sp.csr_matrix((df.shape[0], 0))
        sig_index = {s: j for j, s in enumerate(self.signature_sites_)}
        rows, cols = [], []
        for i, row in df.reset_index(drop=True).iterrows():
            for s in set(self._row_sites(row)):
                j = sig_index.get(s)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        data = np.ones(len(rows), dtype=np.int8)
        return sp.csr_matrix((data, (rows, cols)), shape=(df.shape[0], len(self.signature_sites_)))

    def _row_lengths(self, row: pd.Series) -> np.ndarray:
        vals = [row[c] for c in (self.length_cols_ or []) if c in row and pd.notna(row[c])]
        return pd.to_numeric(pd.Series(vals), errors="coerce").dropna().to_numpy()

    def _session_length_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [
            "total_length",
            "mean_length",
            "median_length",
            "std_length",
            "max_length",
            "min_length",
            "n_sites",
            "n_visits",
        ]
        stats: dict[str, list[float]] = {k: [] for k in cols}

        for _, row in df.iterrows():
            lens = self._row_lengths(row)
            sites = self._row_sites(row)

            if lens.size == 0:
                stats["total_length"].extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                stats["n_sites"].append(float(len(set(sites))))
                stats["n_visits"].append(0.0)
            else:
                stats["total_length"].append(float(np.sum(lens)))
                stats["mean_length"].append(float(np.mean(lens)))
                stats["median_length"].append(float(np.median(lens)))
                stats["std_length"].append(float(np.std(lens)))
                stats["max_length"].append(float(np.max(lens)))
                stats["min_length"].append(float(np.min(lens)))
                stats["n_sites"].append(float(len(set(sites))))
                stats["n_visits"].append(float(lens.size))

        return pd.DataFrame(stats, index=df.index)

    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        dt_date = pd.to_datetime(df[self.date_col], errors="coerce")
        dt_time = pd.to_datetime(df[self.time_col], errors="coerce")
        hour = dt_time.dt.hour.fillna(-1).astype(int)
        out["hour"] = hour
        out["dow"] = dt_date.dt.dayofweek.fillna(-1).astype(int)
        out["month"] = dt_date.dt.month.fillna(-1).astype(int)
        # convert on the fly; do NOT mutate self.night_hours
        night_set = set(self.night_hours)
        out["is_night"] = hour.isin(night_set).astype(int)
        if self.use_cyclic_time:
            hr = hour.clip(lower=0)
            out["hour_sin"] = np.sin(2 * np.pi * hr / 24.0)
            out["hour_cos"] = np.cos(2 * np.pi * hr / 24.0)
        return out


def prepare_dataset_raw(df_raw: pd.DataFrame, cfg: PrepareConfig) -> DatasetBundle:
    if "sites" not in df_raw.columns:
        raise ValueError("Expected column 'sites' in the input DataFrame.")
    k = cfg.max_sites_per_session
    wide = df_raw["sites"].apply(lambda x: sites_to_wide(x, k))
    df_wide = pd.concat([df_raw.drop(columns=["sites"]), wide], axis=1)
    df_wide = normalize_location(df_wide)

    cutoff = pd.to_datetime(cfg.cutoff_date)
    dcol = cfg.date_col
    dt_series = pd.to_datetime(df_wide[dcol], errors="coerce")
    if dt_series.isna().all():
        raise ValueError(f"Column '{dcol}' could not be parsed as dates.")

    mask = dt_series < cutoff
    df_train = df_wide.loc[mask].reset_index(drop=True)
    df_test = df_wide.loc[~mask].reset_index(drop=True)

    if cfg.target_user_col not in df_wide.columns:
        raise ValueError(f"Expected column '{cfg.target_user_col}' for labels.")
    y_train = (df_train[cfg.target_user_col] == cfg.target_user_value).astype(int).to_numpy()
    y_test = (df_test[cfg.target_user_col] == cfg.target_user_value).astype(int).to_numpy()

    # Drop label from raw X to avoid leakage through OHE
    X_train_raw = df_train.drop(columns=[cfg.target_user_col])
    X_test_raw = df_test.drop(columns=[cfg.target_user_col])

    fe = SessionFeatureEngineer(
        target_user_col=cfg.target_user_col,
        target_user_value=cfg.target_user_value,
        max_tfidf_sites=cfg.max_tfidf_sites,
        top_user_signature_sites=cfg.top_user_signature_sites,
        use_cyclic_time=cfg.use_cyclic_time,
        night_hours=cfg.night_hours,  # keep unchanged
        cat_cols=cfg.cat_cols,  # keep unchanged
        date_col=cfg.date_col,
        time_col=cfg.time_col,
    )

    return DatasetBundle(
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        y_train=y_train,
        y_test=y_test,
        df_train=df_train,
        df_test=df_test,
        fe=fe,
        date_col=cfg.date_col,
    )
