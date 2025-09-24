from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from abstract_ranking.two_tower import TTVersion
from ace.configs.config import AppS3Config
from vendor_ranking.two_tower.artefacts_service import (
    CountryServingConfig,
)
from vendor_ranking.two_tower.artefacts_service_registry import VendorArtefactsServiceRegistry
from vendor_ranking.two_tower.model_artifacts import VendorArtifactsManager

if TYPE_CHECKING:
    from pathlib import Path


@pytest_asyncio.fixture
async def s3_artifacts_manager(tmp_path: "Path", s3_app_config: AppS3Config) -> VendorArtifactsManager:
    config: CountryServingConfig = VendorArtefactsServiceRegistry.configs[TTVersion.V22].settings[country := "AE"]
    artifacts_manager = VendorArtifactsManager(
        recall=config.recall,
        base_dir=str(tmp_path / f"inference_model_artifacts/{country}"),
        country=country,
        version=config.version,
        s3_app_config=s3_app_config,
    )
    return artifacts_manager


# noinspection PyMethodMayBeStatic
class ArtifactsManagerTest:
    @pytest.mark.asyncio
    async def test_s3_download_model_artifacts(self, s3_artifacts_manager: "VendorArtifactsManager"):
        await s3_artifacts_manager.download_model_artifacts()
        for path in (
            s3_artifacts_manager.user_model_weights_file_data.local_file_path,
            s3_artifacts_manager.user_model_params_file.local_file_path,
            s3_artifacts_manager.ese_embeddings_file.local_file_path,
        ):
            assert path, "path value is not initialized"
            assert path.exists(), "downloaded file not found"
