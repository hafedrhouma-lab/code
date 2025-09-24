from pathlib import Path
from typing import Optional, Tuple

import mlflow
import structlog
import yaml
from mlflow import MlflowException
from mlflow.entities.model_registry import RegisteredModel, ModelVersion
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, AnyUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LOG: structlog.stdlib.BoundLogger = structlog.get_logger()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='MLFLOW_')
    tracking_uri: AnyUrl = Field(default="http://data.talabat.com/api/public/mlflow")
    web_uri: AnyUrl = Field(default="https://data.talabat.com/services/mlflow")
    metadata_file_path: Path = Field(default=Path("/opt/ml/model/MLmodel"))
    registered_model_meta: Path = Field(default=Path("/opt/ml/model/registered_model_meta"))


SETTINGS = Settings()
CLIENT: Optional[MlflowClient] = None


def get_client() -> MlflowClient:
    global CLIENT
    if not CLIENT:
        CLIENT = MlflowClient(tracking_uri=str(SETTINGS.tracking_uri))
        LOG.info(f"MLFlow Client initialized {CLIENT.tracking_uri}")
    return CLIENT


class RunMetadata(BaseModel):
    run_id: str


class RegisteredModelMeta(BaseModel):
    model_name: str
    model_version: int


def load_registered_model_meta() -> Optional[RegisteredModelMeta]:
    path = SETTINGS.registered_model_meta
    if not path.exists():
        LOG.warning(f"Run metadata file not found: {path}")
        return None

    with open(path, "r") as file:
        data: dict = yaml.safe_load(file)
    return RegisteredModelMeta.parse_obj(data)


def load_run_metadata() -> Optional[RunMetadata]:
    path = SETTINGS.metadata_file_path
    if not path.exists():
        LOG.warning(f"Run metadata file not found: {path}")
        return None

    with open(path, "r") as file:
        data: dict = yaml.safe_load(file)
    return RunMetadata.parse_obj(data)


def _find_model_by_api(meta: RunMetadata) -> Optional[Tuple[RegisteredModel, ModelVersion]]:
    import requests
    url = f"{SETTINGS.tracking_uri}/api/2.0/mlflow/registered-models/search"
    LOG.info(f"Searching for model by API: {url}")
    data: dict = requests.get(url).json()
    for model in data["registered_models"]:
        for item in model["latest_versions"]:
            if item["run_id"] == meta.run_id:
                return (
                    RegisteredModel(name=model["name"]),
                    ModelVersion(name=model["name"], version=int(item["version"]), creation_timestamp=None)
                )
    return None


def find_models(
        meta: RunMetadata,
        client: MlflowClient = None
) -> Optional[tuple[RegisteredModel, ModelVersion]]:
    client = client or get_client()
    try:
        registered_models: list[RegisteredModel] = client.search_registered_models(max_results=100)
    except MlflowException as ex:
        LOG.warning(f"[mlflow {mlflow.__version__}] Error while searching for registered models: {ex}")
        try:
            return _find_model_by_api(meta)
        except Exception as ex:
            LOG.warning(f"Error while searching for registered models API: {ex}")
            return None

    found_models: list[tuple[RegisteredModel, ModelVersion]] = []
    for model in registered_models:
        for version in model.latest_versions:  # type: ModelVersion
            if version.run_id == meta.run_id:
                found_models.append((model, version))

    if not found_models:
        LOG.warning(f"[mlflow {mlflow.__version__}] No models found for run ID: {meta.run_id}")
        return None

    if len(found_models) > 1:
        LOG.warning(f"More than one model found for run ID: {meta.run_id}. Selected the first one.")
    return found_models[0]


def find_model_by_metadata() -> (RegisteredModel, ModelVersion):
    registered_model_meta: "RegisteredModelMeta" = load_registered_model_meta()
    if registered_model_meta:
        LOG.info("Found registered model meta")
        return (
            RegisteredModel(name=registered_model_meta.model_name),
            ModelVersion(name=registered_model_meta.model_name, version=registered_model_meta.model_version, creation_timestamp=None)
        )

    meta: RunMetadata = load_run_metadata()
    LOG.info("Found run metadata")
    if meta:
        found_model = find_models(meta=meta)
        if found_model:
            return found_model
        else:
            return None, None

    LOG.warning("No metadata found. Can not detect model name and version")
    return None, None
