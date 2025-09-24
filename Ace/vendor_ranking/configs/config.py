
from pathlib import Path

from pydantic import BaseModel

from abstract_ranking.config import RankingConfig, TwoTowersServingConfig
from ace.configs.config import AppConfig, StageType
from ace.configs.manager import ConfigManager, EnvAppSettings


class RankingLogicConfig(BaseModel):
    logic: RankingConfig
    two_towers: TwoTowersServingConfig


class VendorRankingConfig(AppConfig):
    ranking: RankingLogicConfig


def load_vendor_ranking_config(stage: StageType = None) -> VendorRankingConfig:
    config: VendorRankingConfig = ConfigManager.load_configuration(
        stage=stage or EnvAppSettings().stage,
        config_type=VendorRankingConfig,
        source_paths=[Path(__file__).parent.resolve()],
    )
    config.ranking.logic.check_correctness()
    return config


VENDOR_RANKING_CONFIG = load_vendor_ranking_config()


def get_vendor_ranking_config() -> VendorRankingConfig:
    global VENDOR_RANKING_CONFIG
    return VENDOR_RANKING_CONFIG
