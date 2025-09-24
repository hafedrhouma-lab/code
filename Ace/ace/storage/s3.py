import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Optional

import aioboto3
import aiohttp.client_exceptions
import botocore.exceptions
import newrelic.agent
import structlog
from aiobotocore.config import AioConfig
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ace.configs.config import AppS3Config
from ace.perf import perf_manager

if TYPE_CHECKING:
    from types_aiobotocore_s3 import S3Client
    from structlog.stdlib import BoundLogger

LOG: "BoundLogger" = structlog.get_logger()


@dataclass
class S3DownloadManager:
    """
    Manager for batch S3 files downloading.
    Examples:
        >>> manager = S3DownloadManager(...)
        >>> future = manager.add_task(
        >>>     "BUCKET_NAME", [(Path("key/in/s3/bucket"), Path("/dst/file/name"))], "just for example")
        >>> await manager.download()
        >>> results = await future
    """

    s3_app_config: AppS3Config
    log_attrs: dict[str, Any] = field(default_factory=dict)
    _key_to_etag: dict[Path, str] = field(default_factory=dict)
    _pending_tasks: dict[asyncio.Future, "S3DownloadManager._Task"] = field(default_factory=dict)

    class _Task(BaseModel):
        bucket: str
        keys_and_dsts: list[tuple[Path, Path]]
        description: str

        @property
        def keys(self) -> list[Path]:
            return [key for key, _ in self.keys_and_dsts]

    def add_task(self, bucket_name: str, keys_and_destinations: list[tuple[Path, Path]], description: str) -> Awaitable:
        """
        Add the task to download files from S3 to local drive.
        That method just creates the task, but doesn't start downloading.

        Args:
            bucket_name:
            keys_and_destinations:
            description:

        Returns: Awaitable object which will contain local paths of downloaded files ones task is done.

        """
        future = asyncio.Future()
        self._pending_tasks[future] = self._Task(
            bucket=bucket_name, keys_and_dsts=keys_and_destinations, description=description
        )
        return future

    @classmethod
    async def _get_s3_etag(cls, bucket: str, key: Path | str, s3_client: "S3Client") -> str:
        try:
            response: dict = await s3_client.head_object(Bucket=bucket, Key=str(key))
            return response["ETag"]
        except botocore.exceptions.ClientError as ex:
            if ex.response["Error"]["Code"] in ("NoSuchKey", "404"):
                raise FileNotFoundError(f"Key NOT FOUND: {bucket}/{key}") from ex

    def clear_pending_tasks(self):
        self._pending_tasks.clear()

    @newrelic.agent.function_trace()
    async def download_batch(
        self, bucket_name: str, keys_and_destinations: list[tuple[Path, Path]], description: str
    ) -> (list[Path], bool):
        """
        Download batch of files with no tasks registration.
        Skip files which S3 ETag was not changed from the previous run.
        """

        task = self._Task(bucket=bucket_name, keys_and_dsts=keys_and_destinations, description=description)
        async with self._create_s3_client() as s3_client:
            paths, reloaded = await self._download_task(task, s3_client=s3_client)  # type: (list[Path], bool)

        return paths, reloaded

    @newrelic.agent.function_trace()
    async def download(self):
        """
        Run downloading for all added tasks.
        Skip files which S3 ETag was not changed from previous run.
        """

        async def __download(
            _task: "S3DownloadManager._Task", _future: asyncio.Future, _s3_client: "S3Client"
        ) -> Optional[list[Path]]:
            try:
                paths, reloaded = await self._download_task(_task, s3_client=_s3_client)  # type: (list[Path], bool)
                _future.set_result((paths, reloaded))
            except Exception as ex:
                _future.set_exception(ex)
            else:
                return paths

        async with self._create_s3_client() as s3_client:
            coros = [
                __download(_task=task, _future=future, _s3_client=s3_client)
                for future, task in self._pending_tasks.items()
            ]
            results = await asyncio.gather(*coros)
            self._pending_tasks.clear()  # futures were done, then they must be cleared from pending tasks
            return results

    async def _download_task(self, task: "S3DownloadManager._Task", s3_client: "S3Client") -> tuple[list[Path], bool]:
        """
        Execute a single task.
        Download files from S3 storage to local drive.
        Skip files which S3 ETag was not changed.

        Args:
            task: which files to download
            s3_client: initialized S3 client

        Returns: List of downloaded local files' paths
                 Flag indicating if any file changed its content

        """
        coros = [
            self._download_file_and_check_etag(bucket=task.bucket, key=key, dst_file_path=dst, s3_client=s3_client)
            for key, dst in task.keys_and_dsts
        ]
        with perf_manager(
            description=f"Finished task `{task.description}`: keys={task.keys}",
            description_before=f"Started task `{task.description}`: keys={task.keys}",
            level=logging.INFO,
            attrs=self.log_attrs,
        ):
            results: list[tuple[Path, bool]] = await asyncio.gather(*coros)
        paths = [path for path, _ in results]
        reloads = [reloaded for _, reloaded in results]
        return paths, any(reloads)

    @retry(
        retry=retry_if_exception_type(aiohttp.client_exceptions.ClientPayloadError),
        stop=stop_after_attempt(3),  # retry 3 time
        wait=wait_fixed(0.1),  # wait 100 milliseconds between retries
    )
    async def _download_file(self, bucket: str, key: Path | str, dst_file_path: Path, s3_client: "S3Client"):
        try:
            dst_file_path.parent.mkdir(parents=True, exist_ok=True)
            result = await s3_client.download_file(bucket, str(key), str(dst_file_path))
            LOG.debug(
                "AWS Downloaded storage object {} from bucket {} to local file {}.".format(key, bucket, dst_file_path),
                **self.log_attrs,
            )
            return result
        except botocore.exceptions.ClientError as ex:
            if ex.response["Error"]["Code"] in ("NoSuchKey", "404"):
                raise FileNotFoundError(f"Key NOT FOUND: {bucket}/{key}") from ex

    async def _download_file_and_check_etag(
        self, bucket: str, key: Path, dst_file_path: Path, s3_client: "S3Client"
    ) -> tuple[Path, bool]:
        reloaded: bool = False
        if not (previous_etag := self._key_to_etag.get(key)):
            current_etag, _ = await asyncio.gather(
                self._get_s3_etag(bucket=bucket, key=key, s3_client=s3_client),
                self._download_file(bucket, key, dst_file_path, s3_client),
            )
            self._key_to_etag[key] = current_etag
            LOG.debug(
                f"New key detected: {key=}, ETag={current_etag}. Initial downloading.",
                **self.log_attrs,
            )
            reloaded = True
        else:
            current_etag: str = await self._get_s3_etag(bucket=bucket, key=key, s3_client=s3_client)
            if current_etag != previous_etag:
                await self._download_file(bucket, key, dst_file_path, s3_client)
                self._key_to_etag[key] = current_etag
                LOG.info(
                    f"Key content has changed: {key=}, ETag={current_etag}. Refresh downloading.",
                    **self.log_attrs,
                )
                reloaded = True
            else:
                LOG.info(
                    f"Key content not changed: {key=}, ETag={current_etag}. Skip downloading.",
                    **self.log_attrs,
                )

        return Path(dst_file_path), reloaded

    def _create_s3_client(self) -> "S3Client":
        session = aioboto3.Session()

        s3_server_port, s3_server_host, aws_access_key_id, aws_secret_access_key = (
            self.s3_app_config.server_port,
            self.s3_app_config.server_host,
            self.s3_app_config.access_key,
            self.s3_app_config.secret_key,
        )
        if all((s3_server_port, s3_server_host, aws_access_key_id, aws_secret_access_key)):
            endpoint_url = f"http://{s3_server_host}:{s3_server_port}"
            LOG.warning(f"Connecting to custom S3 server: url={endpoint_url}", **self.log_attrs)
            s3_client_kwargs = dict(
                endpoint_url=endpoint_url,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
        else:
            s3_client_kwargs = {}

        aioconfig = AioConfig(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
            max_pool_connections=100,
        )
        return session.client("s3", **s3_client_kwargs, config=aioconfig)
