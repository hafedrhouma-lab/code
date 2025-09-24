from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from typing_extensions import Self

from abstract_ranking.two_tower import TTVersion
from ace.enums import CountryShortNameUpper
from menu_item_ranking.artefacts_service_registry import MenuArtefactsServiceRegistry
from menu_item_ranking.user_features.offline.database_provider import DatabaseUserOfflineFeaturesProvider
from menu_item_ranking.user_features.online.provider import UserOnlineFeaturesProvider
from menu_item_ranking.user_features.user_features_manager import UserFeaturesProvider


@dataclass
class UserFeaturesProvidersRegistry:
    _providers: dict[
        TTVersion, dict[CountryShortNameUpper, UserFeaturesProvider]
    ] = field(default_factory=dict)

    def get_provider(
        self, version: TTVersion, country: CountryShortNameUpper
    ) -> UserFeaturesProvider:
        return self._providers[version][country]

    @classmethod
    async def instance(
        cls,
        artifacts_service_registry: MenuArtefactsServiceRegistry,
        active_model_versions: Iterable[TTVersion]
    ) -> Self:
        assert active_model_versions, "active model are empty"
        assert artifacts_service_registry.configs, "artifacts configs are empty"

        online_features_providers: dict[TTVersion, UserOnlineFeaturesProvider] = {
            version: UserOnlineFeaturesProvider(version=version)
            for version in active_model_versions
        }

        features_providers = defaultdict(dict)
        for item_model_version in active_model_versions:
            for country in artifacts_service_registry.configs[item_model_version].countries:
                features_names = list(
                    artifacts_service_registry.get(item_model_version).get_features_names(country)
                )
                online_features_provider = online_features_providers[item_model_version]
                offline_features_provider = await DatabaseUserOfflineFeaturesProvider.instance(
                    version=item_model_version,
                    country=country,
                    artefacts_service_registry=artifacts_service_registry,
                    features_names=features_names
                )

                features_providers[item_model_version][country] = UserFeaturesProvider(
                    online_provider=online_features_provider,
                    offline_provider=offline_features_provider,
                    features_names=features_names
                )
        return cls(_providers=features_providers)
