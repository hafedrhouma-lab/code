# data/io.py
from __future__ import annotations

import json

import pandas as pd


def load_raw_df(path: str) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)
