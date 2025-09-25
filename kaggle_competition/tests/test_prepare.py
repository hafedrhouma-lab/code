# tests/test_prepare.py
from __future__ import annotations

import pandas as pd
from sklearn.base import clone

from data.prepare import (
    SessionFeatureEngineer,
    normalize_location,
    prepare_dataset_raw,
    sites_to_wide,
)


def test_sites_to_wide_and_normalize():
    row = {"sites": [{"site": "a.com", "length": 2}, {"site": "b.com", "length": 5}]}
    wide = sites_to_wide(row["sites"], k=4)
    assert list(wide.index)[:4] == ["site_1", "length_1", "site_2", "length_2"]
    assert wide["site_1"] == "a.com"
    assert pd.isna(wide["site_4"])

    df = pd.DataFrame({"location": ["US/NYC", "FR/Paris", None]})
    df2 = normalize_location(df)
    assert "country" in df2 and "city" in df2
    assert "location" not in df2.columns
    assert df2.loc[0, "country"] == "US"
    assert df2.loc[1, "city"] == "Paris"


def test_session_feature_engineer_cloneable_and_fit_transform(small_raw_df):
    fe = SessionFeatureEngineer(
        target_user_col="user_id",
        target_user_value=0,
        max_tfidf_sites=30,
        top_user_signature_sites=10,
        use_cyclic_time=True,
        night_hours=tuple(range(0, 7)),
        cat_cols=("locale", "country", "city", "browser", "os", "gender"),
        date_col="date",
        time_col="time",
    )

    # Expand minimal inputs to call fit/transform
    # First convert 'sites' to wide to discover columns
    from data.prepare import normalize_location, sites_to_wide

    wide = small_raw_df["sites"].apply(lambda s: sites_to_wide(s, 5))
    df_wide = pd.concat([small_raw_df.drop(columns=["sites"]), wide], axis=1)
    df_wide = normalize_location(df_wide)

    fe2 = clone(fe)  # must not error
    Xt = fe2.fit(df_wide, y=small_raw_df["user_id"]).transform(df_wide)
    assert Xt.shape[0] == df_wide.shape[0]
    assert hasattr(fe2, "feature_names_") and len(fe2.feature_names_) == Xt.shape[1]


def test_prepare_dataset_raw_split_and_no_label_leakage(small_raw_df, small_config):
    bundle = prepare_dataset_raw(small_raw_df, small_config)
    # shapes
    assert bundle.X_train_raw.shape[0] + bundle.X_test_raw.shape[0] == small_raw_df.shape[0]
    assert bundle.y_train.shape[0] == bundle.X_train_raw.shape[0]
    assert "user_id" not in bundle.X_train_raw.columns  # label is dropped
