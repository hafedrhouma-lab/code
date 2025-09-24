import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import newrelic.agent
import pandas as pds
import polars as pl
import pydantic as pd
from pydantic import BaseModel

if TYPE_CHECKING:
    from menu_item_ranking.request.input import MenuItemRequest
    from menu_item_ranking.user_features.offline.abstract_provider import UserOfflineFeaturesProvider
    from menu_item_ranking.user_features.online.abstract_provider import BaseUserOnlineFeaturesProvider


@dataclass
class UserFeaturesProvider:
    """ Online/offline features providers can be treated as a `strategy` pattern. """
    online_provider: "BaseUserOnlineFeaturesProvider"
    offline_provider: "UserOfflineFeaturesProvider"
    features_names: list[str]

    @classmethod
    def instance(
        cls,
        online_provider: "BaseUserOnlineFeaturesProvider",
        offline_provider: "UserOfflineFeaturesProvider",
        features_names: Iterable[str]
    ) -> "UserFeaturesProvider":

        err_validation_msg = (
            f"online and offline user features providers are incompatible: "
            f"online({online_provider.version}) != offline({offline_provider.version})"
        )
        assert online_provider.version is offline_provider.version, err_validation_msg

        return cls(
            online_provider=online_provider,
            offline_provider=offline_provider,
            features_names=list(features_names)
        )

    @classmethod
    def combine_features(
        cls,
        online_features: BaseModel,
        offline_features: BaseModel,
        features_names: list[str]
    ):
        features: dict = (
            online_features.dict(
                by_alias=False,
            ) |
            offline_features.dict(
                by_alias=False
            )
        )
        features_df = pl.from_dicts([features]).to_pandas()
        return features_df[features_names]

    @newrelic.agent.function_trace()
    async def get_features(self, request: "MenuItemRequest") -> pds.DataFrame:
        online_features, offline_features = await asyncio.gather(
            self.online_provider.get_features(request=request),
            self.offline_provider.get_features(request=request),
        )  # type: (pd.BaseModel, pd.BaseModel)
        return self.combine_features(
            online_features=online_features,
            offline_features=offline_features,
            features_names=self.features_names
        )
