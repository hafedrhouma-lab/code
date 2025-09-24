from pathlib import Path

from pydantic import BaseModel

from ace.configs.config import AppConfig, StageType
from ace.configs.manager import ConfigManager, EnvAppSettings


class ExternalAPIs(BaseModel):
    openai_api_key: str
    dh_qcommerce_api_endpoint: str


class SemanticCacheConfig(BaseModel):
    semantic_cache_model: str


class UltronServingConfig(AppConfig):
    external_apis: ExternalAPIs
    openai_cache: SemanticCacheConfig


def load_ultron_serving_config(stage: StageType = None) -> UltronServingConfig:
    config = ConfigManager.load_configuration(
        stage=stage or EnvAppSettings().stage,
        config_type=UltronServingConfig,
        source_paths=[Path(__file__).parent.resolve()],
    )
    return config


ULTRON_SERVING_CONFIG = load_ultron_serving_config()


def get_ultron_serving_config() -> UltronServingConfig:
    global ULTRON_SERVING_CONFIG
    return ULTRON_SERVING_CONFIG
