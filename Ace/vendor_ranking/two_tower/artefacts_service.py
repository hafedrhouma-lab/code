from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Type

import newrelic.agent
import polars as pl
import structlog

from abstract_ranking.two_tower import TTVersion
from abstract_ranking.two_tower.artefacts_service import ArtefactsService, CountryServingConfig
from abstract_ranking.two_tower.names import CHAIN_EMBEDDINGS
from ace.storage import db
from vendor_ranking.two_tower.model_artifacts import VendorArtifactsManager
from vendor_ranking.two_tower.repository.models import UserStaticFeaturesV2, UserStaticFeaturesV3
from vendor_ranking.two_tower.utils import combine_tt_user_features, get_tt_dynamic_features_type

if TYPE_CHECKING:
    import tensorflow as tf
    from structlog.stdlib import BoundLogger

LOG: "BoundLogger" = structlog.get_logger()


@dataclass
class VendorArtefactsServiceBase(ArtefactsService, ABC):

    @property
    def artifacts_manager_type(self) -> Type[VendorArtifactsManager]:
        return VendorArtifactsManager

    @property
    def embeddings_table_name(self):
        return self.get_embeddings_table_name(self.version)

    @classmethod
    def get_embeddings_table_name(cls, version: TTVersion):
        return f"chain_embeddings_for_two_tower_{version}"

    @newrelic.agent.function_trace()
    def _make_warmup_inference(self, model: "tf.keras.Sequential", config: CountryServingConfig):
        from vendor_ranking.two_tower.logic import infer_user_embedding_two_tower

        static_features = self._get_default_user_static_features(config.country)
        dynamic_features_class = get_tt_dynamic_features_type(config.version)
        dynamic_features = dynamic_features_class.from_location(self._default_locations[config.country])

        features_names: set[str] = self.get_features_names(config.country)
        features_df, _ = combine_tt_user_features(dynamic_features, static_features, features_names)
        infer_user_embedding_two_tower(features=features_df, user_model=model)

    def assert_embedding_columns(self, columns: list[str]):
        assert CHAIN_EMBEDDINGS in columns

    async def load_embeddings_from_db(self, country: str) -> "pl.DataFrame":
        """
        Load chain embedding with the next columns:
            chain_id, chain_name, embeddings, country, dwh_entry_date
        """
        query = f"""
            WITH unique_chains AS (
                SELECT DISTINCT ON (chain_id) chain_id, chain_name
                FROM vl_feature_vendors_v3
            )
            SELECT
                chain_embeddings.chain_id,
                unique_chains.chain_name AS chain_name,
                chain_embeddings.embeddings,
                chain_embeddings.country AS country,
                chain_embeddings.dwh_entry_date
            FROM {self.embeddings_table_name} AS chain_embeddings
            LEFT JOIN unique_chains
                ON unique_chains.chain_id = chain_embeddings.chain_id
            WHERE country in ($1, $2)
            ORDER BY chain_name NULLS LAST;
        """
        async with db.connections().acquire() as conn:
            df: "pl.DataFrame" = await db.fetch_as_df(conn, query, country.upper(), country.lower())
        return df


class V2VendorArtefactsService(VendorArtefactsServiceBase):
    """ Suitable for TwoTowers versions V2, V22, V23 """
    def _get_default_user_static_features(
        self, country_code: str, customer_id: int = -1
    ) -> UserStaticFeaturesV2:
        return UserStaticFeaturesV2(
            account_id=customer_id,
            country_iso=country_code,
            most_recent_10_clicks_wo_orders="no_recent_clicks",
            most_recent_10_orders="first_order",
            frequent_clicks="no_frequent_clicks",
            frequent_chains="no_frequent_orders",
            most_recent_15_search_keywords="no_prev_search",
        )


class V3VendorArtefactsService(VendorArtefactsServiceBase):
    """ Suitable for TwoTowers version V3 """
    def _get_default_user_static_features(
        self, country_code: str, customer_id: int = -1
    ) -> UserStaticFeaturesV3:
        return UserStaticFeaturesV3(
            account_id=customer_id,
            country_iso=country_code,
            most_recent_10_clicks_wo_orders="no_recent_clicks",
            most_recent_10_orders="first_order",
            frequent_clicks="no_frequent_clicks",
            frequent_chains="no_frequent_orders",
            most_recent_15_search_keywords="no_prev_search",
            account_order_source="vendor_list",
            account_log_order_cnt=1.6094379124341003,
            account_log_avg_gmv_eur=3.395062551,
            account_incentives_pct=0,
            account_is_tpro=0,
            account_discovery_pct=0.5,
        )
