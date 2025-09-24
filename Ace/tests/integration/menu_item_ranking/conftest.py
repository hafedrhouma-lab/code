from pathlib import Path

import pytest_asyncio

from abstract_ranking.two_tower import TTVersion
from abstract_ranking.two_tower.artefacts_service import CountryServingConfig
from ace.configs.config import AppS3Config
from menu_item_ranking.artefacts_service_registry import MenuArtefactsServiceRegistry
from menu_item_ranking.context import Context as MenuItemsContext
from menu_item_ranking.model_artifacts import MenuItemArtefactsManager


@pytest_asyncio.fixture(scope="function")
async def menu_item_artifacts_service_registry(
    menu_items_app_context: MenuItemsContext
) -> MenuArtefactsServiceRegistry:
    return menu_items_app_context.artifacts_service_registry


@pytest_asyncio.fixture(scope="function")
async def menu_item_artifacts_manager(
    tmp_path: "Path",
    s3_app_config: AppS3Config
) -> MenuItemArtefactsManager:
    config: CountryServingConfig = MenuArtefactsServiceRegistry.configs[TTVersion.MENUITEM_V1].settings[country := "EG"]
    artifacts_manager = MenuItemArtefactsManager(
        recall=config.recall,
        base_dir=str(tmp_path / f"inference_model_artifacts/{country}"),
        country=country,
        version=config.version,
        s3_app_config=s3_app_config,
    )
    return artifacts_manager
