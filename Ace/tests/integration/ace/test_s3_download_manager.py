from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from ace.storage.s3 import S3DownloadManager

if TYPE_CHECKING:
    from ace.configs.config import AppS3Config
    from typing import Awaitable


@pytest_asyncio.fixture
async def s3_download_manager(s3_app_config: "AppS3Config") -> S3DownloadManager:
    return S3DownloadManager(s3_app_config=s3_app_config)


# noinspection PyMethodMayBeStatic
class S3DownloadManagerTest:
    @pytest.mark.asyncio
    async def test_download_one_task_absent_file(
        self, s3_download_manager: "S3DownloadManager", tmp_path: "Path", s3_app_config: "AppS3Config"
    ):
        future: "Awaitable" = s3_download_manager.add_task(
            bucket_name=s3_app_config.bucket_name,
            keys_and_destinations=[("absent_src_file_key", tmp_path / "dst_file_name")],
            description="test downloading",
        )
        await s3_download_manager.download()
        with pytest.raises(FileNotFoundError):
            await future

    @pytest.mark.asyncio
    async def test_download_batch_absent_file(
        self, s3_download_manager: "S3DownloadManager", tmp_path: "Path", s3_app_config: "AppS3Config"
    ):
        with pytest.raises(FileNotFoundError):
            await s3_download_manager.download_batch(
                bucket_name=s3_app_config.bucket_name,
                keys_and_destinations=[("absent_src_file_key", tmp_path / "dst_file_name")],
                description="test downloading",
            )
