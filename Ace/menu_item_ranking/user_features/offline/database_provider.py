from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, ClassVar

import newrelic.agent
import pydantic as pd
import structlog
from fastapi import HTTPException
from starlette import status
from typing_extensions import Self

from abstract_ranking.two_tower import TTVersion
from ace.enums import CountryShortNameUpper
from ace.newrelic import add_transaction_attr
from ace.storage import db
from menu_item_ranking import SERVICE_NAME
from menu_item_ranking.artefacts_service_registry import MenuArtefactsServiceRegistry
from menu_item_ranking.user_features.offline.abstract_provider import AbstractUserOfflineFeaturesProvider
from menu_item_ranking.user_features.offline.features import UserOfflineFeatures

if TYPE_CHECKING:
    from menu_item_ranking.request.input import MenuItemRequest
    from structlog.stdlib import BoundLogger

LOG: "BoundLogger" = structlog.get_logger()


@dataclass
class DatabaseUserOfflineFeaturesProvider(AbstractUserOfflineFeaturesProvider):
    """ Read user features from the postgres database. """
    version: TTVersion
    country: CountryShortNameUpper

    _guest_user_id: ClassVar[int] = -1

    @classmethod
    async def instance(
        cls,
        artefacts_service_registry: MenuArtefactsServiceRegistry,
        version: TTVersion,
        country: CountryShortNameUpper,
        features_names: list[str],
        *args,
        **kwargs
    ) -> Self:
        return cls(version=version, country=country)

    @newrelic.agent.function_trace()
    async def get_user_features_per_account_per_chain(self, conn, request: "MenuItemRequest") -> Optional[UserOfflineFeatures]:
        row: dict | None = await db.fetch_row_as_dict(
            conn,
            self._get_user_features_for_chain_query(),
            request.customer_id,
            self.country,
            request.chain_id
        )
        return row and pd.parse_obj_as(UserOfflineFeatures, row)

    @newrelic.agent.function_trace()
    async def get_user_features_per_account(self, conn, request: "MenuItemRequest") -> UserOfflineFeatures:
        # No previous orders with the chain, using previous orders at other chains
        customer_id = request.customer_id or self._guest_user_id
        rows: list[dict] = await db.fetch_rows_as_dicts(
            conn,
            self._get_user_features_generic_query_with_guest(),
            customer_id,
            self.country,
        )
        row: dict = self.find_row_with_account_id_or_guest(rows, request.customer_id)
        return pd.parse_obj_as(UserOfflineFeatures, row)

    @newrelic.agent.function_trace()
    async def get_features(self, request: "MenuItemRequest") -> UserOfflineFeatures:
        async with db.connections().acquire() as conn:
            if request.customer_id is None:
                return await self.get_user_features_per_account(conn, request)

            if result := await self.get_user_features_per_account_per_chain(conn, request):
                return result

            return await self.get_user_features_per_account(conn, request)

    @newrelic.agent.function_trace()
    def find_row_with_account_id_or_guest(self, rows: list[dict], account_id):
        for row in rows:
            if row.get("account_id") == account_id:
                return row
        # If no account specific data, find guest data
        add_transaction_attr("menuitem.unknown_user", 1, SERVICE_NAME)
        LOG.warning(
            f"User {account_id} not found in {self.country}. Use default guest user features.")

        for row in rows:
            if row.get("account_id") == self._guest_user_id:
                return row

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"CODE": "DEFAULT_USER_STATIC_FEATURES_NOT_FOUND"}
        )

    @classmethod
    def get_users_features_generic_table_name(cls):
        return "inference_features_for_menu_two_tower_v1_per_account"

    @classmethod
    def get_users_features_per_chain_table_name(cls):
        return "inference_features_for_menu_two_tower_v1_per_account_per_chain"

    @staticmethod
    def _get_user_features_for_chain_query():
        return f"""
            SELECT
                per_chain.chain_prev_items as chain_prev_items,
                per_chain.chain_prev_items_names as chain_prev_items_names,
                per_chain.prev_items as prev_items,
                per_chain.prev_items_names as prev_items_names,
                per_chain.freq_items as freq_items,
                per_chain.freq_items_names as freq_items_names
            FROM {DatabaseUserOfflineFeaturesProvider.get_users_features_per_chain_table_name()} per_chain
            WHERE per_chain.account_id = $1 AND per_chain.country_code = $2 AND per_chain.chain_id = $3
        """

    @classmethod
    def _get_user_features_generic_query_with_guest(cls):
        return f"""
            SELECT
                generic.account_id as account_id,
                generic.prev_items as prev_items,
                generic.prev_items_names as prev_items_names,
                generic.freq_items as freq_items,
                generic.freq_items_names as freq_items_names
            FROM {DatabaseUserOfflineFeaturesProvider.get_users_features_generic_table_name()} generic
            WHERE generic.account_id IN($1, {cls._guest_user_id}) AND generic.country_code = $2
        """
