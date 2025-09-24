from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import pandas as pds
import pydantic as pd
from typing_extensions import Self

from abstract_ranking.two_tower import TTVersion
from ace.enums import CountryShortNameUpper
from menu_item_ranking.artefacts_service_registry import MenuArtefactsServiceRegistry
from menu_item_ranking.model_artifacts import MenuItemArtefactsManager
from menu_item_ranking.user_features.offline.abstract_provider import AbstractUserOfflineFeaturesProvider
from menu_item_ranking.user_features.offline.features import UserOfflineFeatures

if TYPE_CHECKING:
    from menu_item_ranking.request.input import MenuItemRequest


@dataclass
class FileUserOfflineFeaturesProvider(AbstractUserOfflineFeaturesProvider):
    """ Read user features from local file downloaded from AWS S3. """
    version: TTVersion
    country: CountryShortNameUpper
    features_names: list[str]
    features: pds.DataFrame
    key_columns: ClassVar[list[str]] = ["account_id", "vendor_id"]

    @classmethod
    async def instance(
        cls,
        artefacts_service_registry: MenuArtefactsServiceRegistry,
        version: TTVersion,
        country: CountryShortNameUpper,
        features_names: list[str],
        **kwargs
    ) -> Self:
        artefacts_service = artefacts_service_registry.artifacts_services[version]
        artifacts_manager: "MenuItemArtefactsManager" = artefacts_service.get_artifacts_manager(country)
        df: "pds.DataFrame" = pds.read_parquet(artifacts_manager.user_features_file.local_file_path)
        df = df[features_names + cls.key_columns]  # select only specific columns
        df = df.astype({key: int for key in cls.key_columns}, copy=True)  # change types
        df.set_index(cls.key_columns, inplace=True)  # create index
        return cls(version=version, country=country, features_names=features_names, features=df)

    async def get_features(self, request: "MenuItemRequest") -> UserOfflineFeatures:
        user_features_set_df: pds.DataFrame = self.features.loc[
            request.customer_id, request.vendor_id
        ]
        user_features_df: pds.Series = user_features_set_df.reset_index().loc[0]
        return pd.parse_obj_as(UserOfflineFeatures, user_features_df)
