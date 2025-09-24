from pathlib import Path

from ace.configs.config import AppConfig, StageType
from ace.configs.manager import ConfigManager, EnvAppSettings


class NBAServingConfig(AppConfig):
    pass


def load_nba_serving_config(stage: StageType = None) -> NBAServingConfig:
    config = ConfigManager.load_configuration(
        stage=stage or EnvAppSettings().stage,
        config_type=NBAServingConfig,
        source_paths=[Path(__file__).parent.resolve()],
    )
    return config


NBA_SERVING_CONFIG = load_nba_serving_config()


def get_nba_serving_config() -> NBAServingConfig:
    global NBA_SERVING_CONFIG
    return NBA_SERVING_CONFIG
