import asyncio
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional, TYPE_CHECKING, TypeVar

import newrelic.agent
import structlog
from keras import Sequential

from abstract_ranking.two_tower import TTVersion
from ace.configs.config import AppS3Config
from ace.storage.s3 import S3DownloadManager

if TYPE_CHECKING:
    pass


logger = structlog.get_logger()


@dataclass
class S3File:
    s3_file_key: str
    local_file_path: Optional[Path] = None


@dataclass
class S3FileBatch:
    description: str
    s3_files: list[S3File]


class ArtefactsManager(ABC):
    """ Utility class which main purpose is to load S3 artifacts for a single model to local storage.
        One model can be uniquely defined by values of the version, country and recall.
    """
    def __init__(
        self,
        s3_app_config: AppS3Config,
        recall: int,
        base_dir: str | Path,
        country: str,
        version: TTVersion
    ):
        self.s3_app_config: AppS3Config = s3_app_config
        self.recall = recall
        self.local_base_path: Path = Path(base_dir)
        self.country = country
        self.version: TTVersion = version

        self._bucket_name: str = s3_app_config.bucket_name
        self.s3_base_path: Path = self._get_s3_base_path(version) / country / f"model_artifacts_recall_{recall}"
        self.s3_downloader = S3DownloadManager(log_attrs=self.artifacts_id, s3_app_config=s3_app_config)

        self.user_model_params_file = S3File(
            s3_file_key=f"tt_user_model_params_recall@10_{recall}.pkl"
        )
        self.user_model_weights_file_data = S3File(
            s3_file_key=f"tt_user_model_weights_recall@10_{recall}.data-00000-of-00001"
        )
        self.model_weights_file_index = S3File(
            s3_file_key=f"tt_user_model_weights_recall@10_{recall}.index"
        )

    @abstractmethod
    def _get_s3_base_path(self, version: TTVersion) -> Path:
        pass

    @abstractmethod
    def instantiate_user_model(self) -> (Sequential, set[str]):
        pass

    async def execute_pending_download_tasks(self):
        await self.s3_downloader.download()

    @newrelic.agent.function_trace()
    async def download_model_artifacts(self, additional_files: list[S3FileBatch]) -> list[bool]:

        files_to_download: list[S3FileBatch] = [self.user_model_files_to_download()]
        if additional_files is not None:
            files_to_download.extend(additional_files)
        futures = []

        for file_batch in files_to_download:
            futures.append(self.s3_downloader.add_task(
                description=f"Downloading {file_batch.description}",
                bucket_name=self._bucket_name,
                keys_and_destinations=[
                    (self.get_src_key_path(file.s3_file_key),
                     self.get_dst_file_path(file.s3_file_key)) for file in file_batch.s3_files
                ],
            ))

        # do actual downloading
        await self.execute_pending_download_tasks()

        # wait and handle results
        results = await asyncio.gather(*futures)

        if len(results) != len(files_to_download):
            raise ValueError(f"Unexpected results format: {results}")

        for idx, (result, download_request) in enumerate(zip(results, files_to_download)):
            downloaded_files, reloaded = result
            if len(download_request.s3_files) != len(downloaded_files):
                raise ValueError(f"Unexpected result length: {downloaded_files}, batch description: {download_request.description}")

            if reloaded:
                for file_idx, result_path in enumerate(downloaded_files):
                    download_request.s3_files[file_idx].local_file_path = result_path

        return [result[1] for result in results]

    @cached_property
    def artifacts_id_repr(self) -> str:
        return f"R10={self.recall}, V={self.version}, C={self.country}"

    @cached_property
    def artifacts_id(self) -> dict:
        return dict(recall=self.recall, version=self.version.value, country=self.country)

    def get_src_key_path(self, filename: str):
        return self.s3_base_path / filename

    def get_dst_file_path(self, filename: str) -> Path:
        return self.local_base_path / filename

    def user_model_files_to_download(self) -> S3FileBatch:
        return S3FileBatch(
            description="user model weights and params",
            s3_files=[
                self.user_model_weights_file_data,
                self.model_weights_file_index,
                self.user_model_params_file,
            ],
        )

    def get_params_file_name(self):
        return self.user_model_params_file

    def get_model_weights_file_name(self):
        return self.user_model_weights_file_data.local_file_path.with_suffix("")

    def load_user_model_params(self) -> dict:
        assert self.user_model_params_file.local_file_path, \
            f"[{self.artifacts_id_repr}] NOT SET: model params file path not set"
        assert (
            self.user_model_params_file.local_file_path.exists()
        ), f"[{self.artifacts_id_repr}] NOT FOUND: model params file {self.user_model_params_file.local_file_path}"
        with open(self.user_model_params_file.local_file_path, "rb") as f:
            experiment_dict = pickle.load(f)

        experiment_dict["account_gmv"] = None
        return experiment_dict


ArtefactsManagerBase = TypeVar("ArtefactsManagerBase", bound=ArtefactsManager)
