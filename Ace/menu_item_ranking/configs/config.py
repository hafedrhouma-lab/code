
from pathlib import Path

from pydantic import BaseModel

from abstract_ranking.config import RankingConfig, TwoTowersServingConfig
from ace.configs.config import AppConfig, StageType
from ace.configs.manager import ConfigManager, EnvAppSettings


class MenuItemRankingLogicConfig(BaseModel):
    logic: RankingConfig
    two_towers: TwoTowersServingConfig


class MenuItemRankingConfig(AppConfig):
    ranking: MenuItemRankingLogicConfig


def load_menu_item_ranking_config(stage: StageType = None) -> MenuItemRankingConfig:
    config: MenuItemRankingConfig = ConfigManager.load_configuration(
        stage=stage or EnvAppSettings().stage,
        config_type=MenuItemRankingConfig,
        source_paths=[Path(__file__).parent.resolve()],
    )
    config.ranking.logic.check_correctness()
    return config


MENU_ITEM_RANKING_CONFIG = load_menu_item_ranking_config()


def get_menu_item_ranking_config() -> MenuItemRankingConfig:
    global MENU_ITEM_RANKING_CONFIG
    return MENU_ITEM_RANKING_CONFIG
