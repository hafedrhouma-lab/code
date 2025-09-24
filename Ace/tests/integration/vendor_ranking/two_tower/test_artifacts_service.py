from typing import TYPE_CHECKING
from unittest import mock

import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

from abstract_ranking.two_tower import TTVersion
from ace.configs.config import AppS3Config
from vendor_ranking.two_tower.artefacts_service import V2VendorArtefactsService, \
    VendorArtefactsServiceBase
from vendor_ranking.two_tower.artefacts_service_registry import VendorArtefactsServiceRegistry

if TYPE_CHECKING:
    from pathlib import Path


@pytest_asyncio.fixture
async def vendor_artifacts_service(tmp_path: "Path", s3_app_config: AppS3Config) -> VendorArtefactsServiceBase:
    with mock.patch(
        "abstract_ranking.two_tower.artefacts_service.get_inference_path",
        new=lambda country: tmp_path / f"inference_model_artifacts/{country}",
    ):
        version = TTVersion.V22
        return V2VendorArtefactsService(
            activated_countries={"AE", "QA"},
            s3_app_config=s3_app_config,
            default_configs=VendorArtefactsServiceRegistry.configs[version].settings,
            version=version
        )


# noinspection PyMethodMayBeStatic
class ArtefactsServiceTest:
    @pytest.mark.asyncio
    async def test_s3_download_all_artifacts_once(self, vendor_artifacts_service: VendorArtefactsServiceBase):
        await vendor_artifacts_service.load()

        for country in vendor_artifacts_service.activated_countries:
            assert vendor_artifacts_service.get_artifacts_manager(country)
            assert vendor_artifacts_service.get_user_model(country)

    @pytest.mark.asyncio
    async def test_s3_download_all_artifacts_twice(
        self, vendor_artifacts_service: VendorArtefactsServiceBase, mocker: MockerFixture
    ):
        # skip models warmup inference as it takes 5-6 seconds and doesn't influence testing feature
        with mock.patch("vendor_ranking.two_tower.artefacts_service.ArtefactsService.make_warmup_inference"):
            # 1: First loading
            await vendor_artifacts_service.load()

            # 2: Setup mocks for methods, which loads models and downloads S3 files
            spies_load_user_model = [
                mocker.spy(vendor_artifacts_service.get_artifacts_manager(country), "instantiate_user_model")
                for country in vendor_artifacts_service.activated_countries
            ]

            spies_download_file = [
                mocker.spy(
                    vendor_artifacts_service.get_artifacts_manager(country).s3_downloader,
                    "_download_file",
                )
                for country in vendor_artifacts_service.activated_countries
            ]

            # 3: Second loading: no filed must be downloaded, no models must be reloaded
            await vendor_artifacts_service.load()

            for spy in spies_download_file:
                spy.assert_not_called()

            for spy in spies_load_user_model:
                spy.assert_not_called()
