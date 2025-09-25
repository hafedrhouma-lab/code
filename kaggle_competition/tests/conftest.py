# tests/conftest.py
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from data.prepare import PrepareConfig, prepare_dataset_raw


@pytest.fixture(scope="session")
def rng():
    return np.random.RandomState(123)


def _make_sites_row(rng, k=5) -> list[dict]:
    # Random-ish sites with short "length" values
    sites = []
    for _i in range(k):
        sites.append({"site": f"site_{rng.randint(0, 20)}.com", "length": int(rng.randint(0, 8))})
    return sites


def _make_dates(n, start="2018-01-01"):
    start_dt = datetime.fromisoformat(start)
    return [(start_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n)]


def _make_times(n):
    # HH:MM format cycling 0..23
    return [f"{(i % 24):02d}:00:00" for i in range(n)]


def make_raw_df(n=200, k=5, pos_ratio=0.2, rng=None) -> pd.DataFrame:
    rng = rng or np.random.RandomState(123)
    users = np.where(rng.rand(n) < pos_ratio, 0, 1)  # target_user_value=0 is positive
    data = {
        "user_id": users,
        "location": [f"US/City{rng.randint(0, 5)}" for _ in range(n)],
        "browser": rng.choice(["chrome", "firefox", "safari"], size=n),
        "os": rng.choice(["mac", "win", "linux"], size=n),
        "gender": rng.choice(["M", "F"], size=n),
        "locale": rng.choice(["en_US", "fr_FR"], size=n),
        "date": _make_dates(n),
        "time": _make_times(n),
        "sites": [_make_sites_row(rng, k=k) for _ in range(n)],
        "session_id": np.arange(n),
    }
    return pd.DataFrame(data)


@pytest.fixture()
def small_config() -> PrepareConfig:
    return PrepareConfig(
        max_sites_per_session=5,
        cutoff_date="2019-01-01",
        target_user_col="user_id",
        target_user_value=0,
        max_tfidf_sites=50,
        top_user_signature_sites=20,
        use_cyclic_time=True,
        night_hours=tuple(range(0, 7)),
        cat_cols=("locale", "country", "city", "browser", "os", "gender"),
        date_col="date",
        time_col="time",
    )


@pytest.fixture()
def small_raw_df(rng) -> pd.DataFrame:
    return make_raw_df(n=120, k=5, pos_ratio=0.3, rng=rng)


@pytest.fixture()
def small_bundle(small_raw_df, small_config):
    return prepare_dataset_raw(small_raw_df, small_config)
