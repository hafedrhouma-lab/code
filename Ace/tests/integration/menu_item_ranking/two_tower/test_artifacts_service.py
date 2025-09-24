from typing import TYPE_CHECKING
from unittest import mock

import pytest_asyncio

from abstract_ranking.two_tower import TTVersion
from ace.configs.config import AppS3Config
from menu_item_ranking.artefacts_service import MenuItemArtefactsService
from menu_item_ranking.artefacts_service_registry import MenuArtefactsServiceRegistry
from vendor_ranking.two_tower.artefacts_service import ArtefactsService

if TYPE_CHECKING:
    from pathlib import Path


@pytest_asyncio.fixture
async def s3_artifacts_service(tmp_path: "Path", s3_app_config: AppS3Config) -> ArtefactsService:
    with mock.patch(
        "abstract_ranking.two_tower.artefacts_service.get_inference_path",
        new=lambda country: tmp_path / f"inference_model_artifacts/{country}",
    ):
        version = TTVersion.MENUITEM_V1
        return MenuItemArtefactsService(
            activated_countries={"EG"},
            s3_app_config=s3_app_config,
            default_configs=MenuArtefactsServiceRegistry.configs[version].settings,
            version=version
        )


# noinspection PyMethodMayBeStatic
class ArtefactsServiceTest:
    # TODO re-introduce tests once database structure is known
    # @pytest.mark.asyncio
    # async def test_s3_download_all_artifacts_once(self, s3_artifacts_service: "ArtefactsService"):
    #     await s3_artifacts_service.load()
    #
    #     for country in s3_artifacts_service.activated_countries:
    #         assert s3_artifacts_service.get_artifacts_manager(country)
    #         assert s3_artifacts_service.get_user_model(country)
    #
    # @pytest.mark.asyncio
    # async def test_s3_download_all_artifacts_twice(
    #     self, s3_artifacts_service: "ArtefactsService", mocker: MockerFixture
    # ):
    #     # skip models warmup inference as it takes 5-6 seconds and doesn't influence testing feature
    #     with mock.patch("vendor_ranking.two_tower.artefacts_service.ArtefactsService.make_warmup_inference"):
    #         # 1: First loading
    #         await s3_artifacts_service.load()
    #
    #         # 2: Setup mocks for methods, which loads models and downloads S3 files
    #         spies_load_user_model = [
    #             mocker.spy(s3_artifacts_service.get_artifacts_manager(country), "instantiate_user_model")
    #             for country in s3_artifacts_service.activated_countries
    #         ]
    #
    #         spies_download_file = [
    #             mocker.spy(
    #                 s3_artifacts_service.get_artifacts_manager(country).s3_downloader,
    #                 "_download_file",
    #             )
    #             for country in s3_artifacts_service.activated_countries
    #         ]
    #
    #         # 3: Second loading: no filed must be downloaded, no models must be reloaded
    #         await s3_artifacts_service.load()
    #
    #         for spy in spies_download_file:
    #             spy.assert_not_called()
    #
    #         for spy in spies_load_user_model:
    #             spy.assert_not_called()
    pass
