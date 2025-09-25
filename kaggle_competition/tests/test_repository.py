# tests/test_repository.py
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from data.prepare import SessionFeatureEngineer
from model.repository import ArtifactRepository


def test_repository_roundtrip(tmp_path, small_raw_df):
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
    clf = LogisticRegression(max_iter=200)
    pipe = Pipeline([("fe", fe), ("clf", clf)])

    # small fit
    from data.prepare import normalize_location, sites_to_wide

    wide = small_raw_df["sites"].apply(lambda s: sites_to_wide(s, 5))
    df_wide = normalize_location(small_raw_df.drop(columns=["sites"]).join(wide))
    y = (small_raw_df["user_id"] == 0).astype(int).to_numpy()
    pipe.fit(df_wide, y)

    repo = ArtifactRepository()
    path = tmp_path / "artifact.joblib"
    repo.save(str(path), pipe, threshold=0.42)

    art = repo.load(str(path))
    assert art.threshold == 0.42
    assert hasattr(art.pipeline, "named_steps")
    assert art.model is not None and art.fe is not None
