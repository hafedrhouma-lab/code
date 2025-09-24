import asyncio
import logging
from abc import ABC, abstractmethod
from abc import ABCMeta
from typing import ClassVar, TYPE_CHECKING, Callable
from typing import Optional

import newrelic.agent
import numpy as np
import pandas as pds
import polars as pl
import structlog
from fastapi import HTTPException
from starlette import status
from structlog.stdlib import BoundLogger

from abstract_ranking.base_logic import BaseLogic
from abstract_ranking.two_tower import TTVersion
from abstract_ranking.two_tower.names import (
    ORIGINAL_RANK, MENU_ITEM_EMBEDDINGS,
    COSINE_SIMILARITY, MENU_ITEM_RANK, CHAIN_ID, MENU_SOURCE_SYSTEM_ITEM_ID
)
from ace import ml
from ace.enums import CountryShortNameUpper
from ace.model_log import LogEntry
from ace.perf import perf_manager
from ace.storage.db import assert_table_exists
from menu_item_ranking import SERVICE_NAME
from menu_item_ranking.artefacts_service import MenuItemArtefactsService
from menu_item_ranking.request.input import MenuItemRequest
from menu_item_ranking.user_features.offline.database_provider import DatabaseUserOfflineFeaturesProvider
from menu_item_ranking.user_features.offline.features import UserOfflineFeatures
from menu_item_ranking.user_features.online.features import UserOnlineFeatures
from menu_item_ranking.user_features.user_features_manager import UserFeaturesProvider

if TYPE_CHECKING:
    from abstract_ranking.two_tower.artefacts_service import ArtefactsService, CountryServingConfig
    from menu_item_ranking.context import Context as MenuItemRankingContext
    import tensorflow as tf

LOG: "BoundLogger" = structlog.get_logger()


class MenuItemBaseLogic(BaseLogic["MenuItemRequest"], ABC):
    NAME: ClassVar[str] = SERVICE_NAME
    SERVICE_NAME: ClassVar[str] = SERVICE_NAME
    MODEL_TAG: ClassVar[str] = f"{SERVICE_NAME}_date"  # TODO: Change date when real model will be deployed
    VERSION: ClassVar[str | None] = None

    def __init__(self, request: MenuItemRequest, exec_log: "LogEntry", context: "MenuItemRankingContext", **kwargs):
        super().__init__(request, exec_log)
        self.context: "MenuItemRankingContext" = context

    @abstractmethod
    async def prepare_features(self) -> None:
        pass

    @abstractmethod
    async def sort(self) -> list[int]:
        pass


class MenuItemLogic(MenuItemBaseLogic, metaclass=ABCMeta):
    VERSION: ClassVar[TTVersion] = TTVersion.MENUITEM_V1

    def __init__(self, request: MenuItemRequest, exec_log: "LogEntry", context: "MenuItemRankingContext", **kwargs):
        super().__init__(request, exec_log, context)
        self.sorted_menu_items_similarity_scores: Optional[pl.DataFrame] = None
        self._menu_items_with_ranks: Optional[pl.DataFrame] = None

        if (
            request.location.country_code
            not in self.context.artifacts_service_registry.get(self.VERSION).activated_countries
        ):
            raise HTTPException(
                status_code=status.HTTP_406_NOT_ACCEPTABLE,
                detail={
                    "code": "COUNTRY_NOT_SUPPORTED_FOR_VERSION",
                    "description": f"country not supported: {request.location.country_code}, {self.VERSION}",
                },
            )

    @newrelic.agent.function_trace()
    async def prepare_features(self) -> None:
        country: "CountryShortNameUpper" = self.request.location.country_code

        user_features: "pds.DataFrame" = await self._collect_user_features(country=country)
        embedding = await asyncio.to_thread(
            self._infer_user_embedding, country=country, features=user_features, msg=""
        )

        return await asyncio.to_thread(self._prepare_menu_items_ranks, embedding)

    @newrelic.agent.function_trace()
    async def sort(self) -> list[int]:
        err_msg = "items' ranks are not ready, please calculate them"
        assert not (self._menu_items_with_ranks is None or self._menu_items_with_ranks.is_empty()), err_msg

        with perf_manager(
            f"Sorted {self._menu_items_with_ranks.height} items",
            level=logging.DEBUG,
        ):
            sorted_menu_items: "pl.DataFrame" = self._menu_items_with_ranks.sort(
                COSINE_SIMILARITY, ORIGINAL_RANK, descending=[True, False]
            )
            return sorted_menu_items[MENU_SOURCE_SYSTEM_ITEM_ID].to_list()

    @classmethod
    async def check_execution_requirements(cls, msg: str):
        """- Postgres DB must contain not empty tables with TwoTower users' and chains' features and embeddings.
        - Models must be present in S3 for all required countries and recall values.
        """
        await asyncio.gather(
            *(
                assert_table_exists(table_name=table_name, msg=msg)
                for table_name in (
                    MenuItemArtefactsService.get_embeddings_table_name(cls.VERSION),
                    DatabaseUserOfflineFeaturesProvider.get_users_features_generic_table_name(),
                    DatabaseUserOfflineFeaturesProvider.get_users_features_per_chain_table_name()
                )
            )
        )

    @classmethod
    @newrelic.agent.function_trace()
    def call_user_model(
        cls, features: "pds.DataFrame", user_model: "tf.keras.Sequential"
    ):
        features = {
            feature_col: np.asarray(features[feature_col])
            for feature_col in features.columns
        }
        out = features
        for layer in user_model.layers:
            inp = out
            out = layer(inp)
        return out.numpy()[0]

    @classmethod
    @newrelic.agent.function_trace()
    def call_user_model_tf_func(
        cls,
        features: "pds.DataFrame",
        user_model_tf_func: "Callable",
        user_model: "tf.keras.Sequential"
    ):
        _features = {feature_name: np.asarray(features[feature_name]) for feature_name in features.columns}
        return user_model_tf_func(user_model, _features).numpy()

    @newrelic.agent.function_trace()
    def _infer_user_embedding(self, country: "CountryShortNameUpper", features: "pds.DataFrame", msg: str = ""):
        artefacts_service: "ArtefactsService" = self.context.artifacts_service_registry.get(self.VERSION)
        user_model_tf_func = artefacts_service.get_tf_func_user_model(country=country)
        user_model = artefacts_service.get_user_model(country=country)

        config: "CountryServingConfig" = self.context.artifacts_service_registry.get_config(
            self.VERSION, country
        )

        with perf_manager(
            f"{msg}Model inference: {self.VERSION}, {country}, {config.recall}",
            level=logging.DEBUG,
        ):
            embeddings = self.call_user_model_tf_func(
                features=features,
                user_model_tf_func=user_model_tf_func,
                user_model=user_model
            )
        return embeddings[0]

    @newrelic.agent.function_trace()
    async def _collect_user_features(self, country: "CountryShortNameUpper") -> "pds.DataFrame":
        user_features_provider: UserFeaturesProvider = self.context.user_features_providers_registry.get_provider(
            version=self.VERSION, country=country
        )
        with perf_manager(f"Collected features for user {self.request.customer_id}"):
            user_features: "pds.DataFrame" = await user_features_provider.get_features(request=self.request)
        LOG.debug(f"User features: {user_features.to_dict()}")
        return user_features

    @newrelic.agent.function_trace()
    def _collect_default_user_features(self, country: "CountryShortNameUpper") -> "pds.DataFrame":
        user_features_provider: UserFeaturesProvider = self.context.user_features_providers_registry.get_provider(
            version=self.VERSION, country=country
        )
        default_offline_features = UserOfflineFeatures()
        online_features = UserOnlineFeatures.from_request(self.request)

        user_features: "pds.DataFrame" = user_features_provider.combine_features(
            online_features=online_features,
            offline_features=default_offline_features,
            features_names=user_features_provider.features_names
        )
        LOG.debug(f"Default user features: {user_features.to_dict()}")
        return user_features

    @newrelic.agent.function_trace()
    def _prepare_menu_items_ranks(self, user_embedding: np.ndarray):
        requested_chain_id: int = self._get_chain_id()
        menu_item_embeddings: "pl.DataFrame" = self._get_menu_item_embeddings(
            country=self.request.location.country_code,
            chain_id=requested_chain_id
        )
        with perf_manager(
            f"Build scores for {menu_item_embeddings.height} items in chain {requested_chain_id}",
            level=logging.DEBUG,
        ):
            self._menu_items_with_ranks: "pl.DataFrame" = self._build_similarity_scores(
                menu_item_embeddings, user_embedding
            )

        # TODO provide logging
        # self.stats += [
        #     ("2towers.request.total_chains_cnt", all_chains_cnt),
        #     ("2towers.request.avail_chains_cnt", avail_chains_cnt),
        #     (
        #         "2towers.request.avail_chains_pct",
        #         (avail_chains_cnt / all_chains_cnt) * 100,
        #     ),
        #     ("2towers.request.avg_cosine_similarity", avg_cosine_similarity),
        # ]
        # self.exec_log.add_metadata("total_chains_cnt", all_chains_cnt)
        # self.exec_log.add_metadata("avail_chains_cnt", avail_chains_cnt)
        # self.exec_log.add_metadata(
        #     "avail_chains_pct", (avail_chains_cnt / all_chains_cnt) * 100
        # )

        return self._menu_items_with_ranks

    def _get_chain_id(self) -> int:
        if self.request.chain_id is None:
            raise HTTPException(
                status_code=status.HTTP_406_NOT_ACCEPTABLE,
                detail={
                    "code": "CHAIN_ID_CANNOT_BE_NULL",
                    "description": f"chain_id parameter is required.",
                },
            )
        return self.request.chain_id

    @classmethod
    @newrelic.agent.function_trace()
    def _build_similarity_scores(cls, menu_items: "pl.DataFrame", customer_embedding: np.ndarray) -> "pl.DataFrame":
        menu_items_with_embeddings = menu_items.filter(
            pl.col(MENU_ITEM_EMBEDDINGS).is_not_null()
        )
        if menu_items_with_embeddings.height == 0:
            return pl.DataFrame(schema=menu_items.schema).with_columns(
                [
                    pl.lit(None).alias(name)
                    for name in (
                        COSINE_SIMILARITY,
                        MENU_ITEM_RANK,
                    )
                ]
            )

        # get cosine similarities between user and menu_items
        menu_items_with_embeddings = ml.with_cosine_similarity(
            menu_items_with_embeddings,
            customer=customer_embedding,
            embeddings_column_name=MENU_ITEM_EMBEDDINGS,
        )

        return menu_items_with_embeddings

    @newrelic.agent.function_trace()
    def _get_menu_item_embeddings(
        self, country: CountryShortNameUpper | str, chain_id: int
    ) -> "pl.DataFrame":
        items_embeddings_for_country: "pl.DataFrame" = (
            self.context.artifacts_service_registry.get(self.VERSION).get_embeddings(country)
        )
        items_embeddings_for_chain: "pl.DataFrame" = items_embeddings_for_country.filter(
            pl.col(CHAIN_ID).eq(chain_id)
        )
        menu_items_with_original_rank = items_embeddings_for_chain.with_columns(
            pl.arange(start=0, end=items_embeddings_for_chain.height).alias(ORIGINAL_RANK)
        )
        return menu_items_with_original_rank
