from pathlib import Path

from ace.configs.config import AppConfig, StageType
from ace.configs.manager import ConfigManager, EnvAppSettings


class ItemLifecycleServingConfig(AppConfig):
    pass


def load_item_lifecycle_serving_config(stage: StageType = None) -> ItemLifecycleServingConfig:
    config = ConfigManager.load_configuration(
        stage=stage or EnvAppSettings().stage,
        config_type=ItemLifecycleServingConfig,
        source_paths=[Path(__file__).parent.resolve()],
    )
    return config


ITEM_LIFECYCLE_SERVING_CONFIG = load_item_lifecycle_serving_config()


def get_item_lifecycle_serving_config() -> ItemLifecycleServingConfig:
    global ITEM_LIFECYCLE_SERVING_CONFIG
    return ITEM_LIFECYCLE_SERVING_CONFIG
