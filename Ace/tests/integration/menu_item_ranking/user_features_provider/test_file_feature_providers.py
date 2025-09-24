from typing import TYPE_CHECKING

import pandas as pds
import pytest
import pytest_asyncio

from abstract_ranking.two_tower import TTVersion
from ace.enums import CountryShortNameUpper
from menu_item_ranking.artefacts_service_registry import MenuArtefactsServiceRegistry
from menu_item_ranking.request.input import MenuItemRequest
from menu_item_ranking.user_features.offline.abstract_provider import UserOfflineFeaturesProvider
from menu_item_ranking.user_features.offline.database_provider import DatabaseUserOfflineFeaturesProvider
from menu_item_ranking.user_features.online.abstract_provider import BaseUserOnlineFeaturesProvider
from menu_item_ranking.user_features.online.provider import UserOnlineFeaturesProvider
from menu_item_ranking.user_features.user_features_manager import UserFeaturesProvider

if TYPE_CHECKING:
    from abstract_ranking.two_tower.artefacts_service import ArtefactsService


@pytest.fixture(scope="function")
def user_online_features_provider(item_model_version: TTVersion) -> BaseUserOnlineFeaturesProvider:
    return UserOnlineFeaturesProvider(version=item_model_version)


@pytest_asyncio.fixture(scope="function")
async def file_user_offline_features_provider(
    item_model_version: TTVersion,
    menu_item_artifacts_service_registry: MenuArtefactsServiceRegistry,
    item_model_country: CountryShortNameUpper
) -> UserOfflineFeaturesProvider:

    art_service: "ArtefactsService" = menu_item_artifacts_service_registry.get(item_model_version)
    features_names = art_service.get_features_names(item_model_country)

    return await DatabaseUserOfflineFeaturesProvider.instance(
        version=item_model_version,
        country=item_model_country,
        artefacts_service_registry=menu_item_artifacts_service_registry,
        features_names=list(features_names)
    )


@pytest_asyncio.fixture(scope="function")
async def file_user_features_provider(
    user_online_features_provider: BaseUserOnlineFeaturesProvider,
    file_user_offline_features_provider: UserOfflineFeaturesProvider,
    menu_item_artifacts_service_registry: MenuArtefactsServiceRegistry,
    item_model_version: TTVersion,
    item_model_country: CountryShortNameUpper
) -> UserFeaturesProvider:
    art_service: "ArtefactsService" = menu_item_artifacts_service_registry.get(item_model_version)
    return UserFeaturesProvider.instance(
        online_provider=user_online_features_provider,
        offline_provider=file_user_offline_features_provider,
        features_names=art_service.get_features_names(item_model_country)
    )


# noinspection PyMethodMayBeStatic
class FileUserFeaturesProviderTest:
    @pytest.mark.asyncio
    async def test_user_features_calculation(
        self,
        file_user_features_provider: UserFeaturesProvider,
        menu_item_ranking_request: MenuItemRequest
    ):
        features_df: "pds.DataFrame" = await file_user_features_provider.get_features(
            request=menu_item_ranking_request
        )
        assert features_df is not None
        assert features_df.shape == (1, 9)
        features = dict(features_df.loc[0])
        assert features.pop("order_hour") is not None
        assert features.pop("order_weekday") is not None
        assert features == {
            "freq_items": "item1 item2 item3",
            "prev_items_names": "prev_item1_name prev_item2_name",
            "freq_items_names": "item1_name item2_name item3_name",
            "prev_items": "prev_item1 prev_item2",
            "delivery_area_id": 1272,
            "chain_prev_items": "discovery_order",
            "chain_prev_items_names": "discovery_order",
        }
