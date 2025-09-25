# tests/test_infer_prep.py
from __future__ import annotations

from data.prepare import SessionFeatureEngineer
from model.infer.prep import expand_like_training, labels_from_df


def test_expand_like_training_creates_expected_columns(small_raw_df):
    fe = SessionFeatureEngineer(
        target_user_col="user_id",
        target_user_value=0,
        max_tfidf_sites=10,
        top_user_signature_sites=5,
        use_cyclic_time=True,
        night_hours=tuple(range(0, 7)),
        cat_cols=("locale", "country", "city", "browser", "os", "gender"),
        date_col="date",
        time_col="time",
    )

    df_inf = expand_like_training(
        small_raw_df[
            ["sites", "location", "browser", "os", "gender", "locale", "date", "time"]
        ].copy(),
        fe,
    )
    # site and length columns must exist
    assert any(c.startswith("site_") for c in df_inf.columns)
    assert any(c.startswith("length_") for c in df_inf.columns)
    # normalized location columns exist
    assert "country" in df_inf.columns and "city" in df_inf.columns
    # cat and time columns exist
    for c in list(fe.cat_cols) + [fe.date_col, fe.time_col]:
        assert c in df_inf.columns


def test_labels_from_df(small_raw_df):
    fe = SessionFeatureEngineer(
        target_user_col="user_id",
        target_user_value=0,
        max_tfidf_sites=10,
        top_user_signature_sites=5,
        use_cyclic_time=True,
        night_hours=tuple(range(0, 7)),
        cat_cols=("locale", "country", "city", "browser", "os", "gender"),
        date_col="date",
        time_col="time",
    )
    y = labels_from_df(small_raw_df, fe)
    assert y is not None and y.shape[0] == small_raw_df.shape[0]
