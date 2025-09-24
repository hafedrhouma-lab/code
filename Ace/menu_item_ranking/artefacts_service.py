import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Type, Callable, ClassVar

import newrelic.agent
import polars as pl
import structlog

from abstract_ranking.two_tower import TTVersion
from abstract_ranking.two_tower.artefacts_service import ArtefactsService
from abstract_ranking.two_tower.artefacts_service import CountryServingConfig
from abstract_ranking.two_tower.names import (
    MENU_ITEM_ID, MENU_ITEM_EMBEDDINGS, CHAIN_ID, COUNTRY_CODE,
    MENU_SOURCE_SYSTEM_ITEM_ID
)
from ace.perf import perf_manager
from ace.storage import db
from menu_item_ranking.model_artifacts import MenuItemArtefactsManager
from menu_item_ranking.user_features import get_items_tt_user_online_features_type
from menu_item_ranking.user_features.offline.features import UserOfflineFeatures
from menu_item_ranking.user_features.user_features_manager import UserFeaturesProvider

if TYPE_CHECKING:
    import tensorflow as tf
    import pydantic as pd
    from structlog.stdlib import BoundLogger
    from abstract_ranking.two_tower.artefacts_service import CountryServingConfig


LOG: "BoundLogger" = structlog.get_logger()


@dataclass
class MenuItemArtefactsService(ArtefactsService):
    REQUIRED_COLS: ClassVar[set[str]] = {
        MENU_ITEM_EMBEDDINGS, MENU_ITEM_ID, MENU_SOURCE_SYSTEM_ITEM_ID, CHAIN_ID, COUNTRY_CODE
    }

    @property
    def artifacts_manager_type(self) -> Type[MenuItemArtefactsManager]:
        return MenuItemArtefactsManager

    @newrelic.agent.function_trace()
    def make_warmup_inference(self):
        for country_code, model in self._user_models.items():
            # need to measure only one specific call, and newrelic not that much useful here
            serving_config = self.default_configs.get(country_code)
            with perf_manager(
                f"Menu Items Warmup inference for TwoTowers model [{serving_config}] DONE",
                level=logging.INFO
            ):
                self._make_warmup_inference(
                    model_caller=self.get_tf_func_user_model(country=country_code),
                    model=model,
                    config=serving_config
                )

    @newrelic.agent.function_trace()
    def _make_warmup_inference(
        self,
        model_caller: "Callable",
        model: "tf.keras.Sequential",
        config: "CountryServingConfig"
    ):
        offline_features: "pd.BaseModel" = self._get_default_user_static_features(config.country)
        online_features_class = get_items_tt_user_online_features_type(config.version)
        online_features: "pd.BaseModel" = online_features_class.from_location(
            self._default_locations[config.country]
        )

        features_names: list[str] = list(self.get_features_names(config.country))
        features_df = UserFeaturesProvider.combine_features(
            online_features=online_features, offline_features=offline_features, features_names=features_names
        )

        from menu_item_ranking.logic.base_logic import MenuItemLogic
        MenuItemLogic.call_user_model_tf_func(
            features=features_df,
            user_model_tf_func=model_caller,
            user_model=model
        )

    @property
    def embeddings_table_name(self):
        return MenuItemArtefactsService.get_embeddings_table_name(self.version)

    @classmethod
    def get_embeddings_table_name(cls, version: TTVersion = TTVersion.MENUITEM_V1) -> str:
        return f"item_embeddings_for_menu_two_tower_{version}"

    @classmethod
    def assert_embedding_columns(cls, columns: list[str]):
        db_columns: set[str] = set(columns)
        lost_cols = cls.REQUIRED_COLS.difference(db_columns)
        assert not lost_cols, f"columns {lost_cols} not found in db"

    def assert_features_columns(self, columns: list[str]):
        db_columns: set[str] = set(columns)
        for country, serving_config in self.default_configs.items():
            required_features: set[str] = self.get_features_names(country)
            lost_features = required_features.difference(db_columns)
            assert lost_features, f"features {lost_features} for {serving_config} not found in db"

    async def load_embeddings_from_db(self, country: str) -> "pl.DataFrame":
        """ Load items embeddings """
        query = f"""
            SELECT
                item_embeddings.item_id as {MENU_ITEM_ID},
                item_embeddings.source_system_item_id as {MENU_SOURCE_SYSTEM_ITEM_ID},
                item_embeddings.chain_id as {CHAIN_ID},
                item_embeddings.country_code AS {COUNTRY_CODE},
                item_embeddings.item_embeddings AS {MENU_ITEM_EMBEDDINGS}
            FROM {self.embeddings_table_name} AS item_embeddings
            WHERE country_code in ($1, $2);
        """
        async with db.connections().acquire() as conn:
            df: "pl.DataFrame" = await db.fetch_as_df(conn, query, country.upper(), country.lower())
        return df

    def _get_default_user_static_features(self, country_code: str, customer_id: int = -1) -> "pd.BaseModel":
        return UserOfflineFeatures()
