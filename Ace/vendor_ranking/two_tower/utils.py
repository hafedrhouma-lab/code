from typing import TYPE_CHECKING, Union

import pandas as pd
import polars as pl

from abstract_ranking.two_tower import TTVersion
from vendor_ranking.two_tower.repository.models import UserDynamicFeatures, UserDynamicFeaturesV3

if TYPE_CHECKING:
    from vendor_ranking.two_tower.repository.models import (
        UserStaticFeaturesV2, UserStaticFeaturesV3
)


def combine_tt_user_features(
    dynamic_features: "UserDynamicFeatures",
    static_features: Union["UserStaticFeaturesV2", "UserStaticFeaturesV3"],
    features_names: set[str]
) -> (pd.DataFrame, dict):
    features: dict = (
        static_features.dict(
            by_alias=True,
        ) |
        dynamic_features.dict(
            by_alias=False
        )
    )
    features_df = pl.from_dicts([features]).to_pandas()
    return features_df[list(features_names)], features


def get_tt_dynamic_features_type(version: TTVersion):
    if version in (TTVersion.V2, TTVersion.V22, TTVersion.V23):
        return UserDynamicFeatures
    if version is TTVersion.V3:
        return UserDynamicFeaturesV3
    raise ValueError(f"Unsupported model version {version}")
