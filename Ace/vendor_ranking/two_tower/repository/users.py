import operator
from typing import TYPE_CHECKING

import asyncache
import cachetools
import newrelic.agent
import pydantic
import structlog
from cachetools import Cache
from cachetools.keys import methodkey
from fastapi import HTTPException
from starlette import status

from ace.newrelic import add_transaction_attr
from ace.storage import db
from vendor_ranking import SERVICE_NAME
from abstract_ranking.two_tower import TTVersion
from vendor_ranking.two_tower.repository.models import UserStaticFeaturesV2, UserStaticFeaturesV3

if TYPE_CHECKING:
    import polars as pl
    from structlog.stdlib import BoundLogger

LOG: "BoundLogger" = structlog.get_logger()


class TwoTowersUsersRepository:
    def __init__(self, version: TTVersion = TTVersion.V22):
        self.version: TTVersion = version
        if version in (TTVersion.V2, TTVersion.V22, TTVersion.V23):
            self.features_type = UserStaticFeaturesV2
        elif version is TTVersion.V3:
            self.features_type = UserStaticFeaturesV3
        else:
            raise ValueError(f"Unsupported model version {version}")

        self.users_features_table_name: str = f"inference_features_for_two_tower_{version}"
        self._user_features_query: str = (
            f"SELECT * FROM {self.users_features_table_name} WHERE account_id = $1 AND {{country_column}} = $2"
        )
        self._guest_user_features_query: str = (
            f"SELECT * FROM {self.users_features_table_name} WHERE account_id = -1 AND {{country_column}} = $1"
        )
        self._guest_users_cache = Cache(100)
        self._user_query_cache = Cache(100)
        self._guest_user_query_cache = Cache(100)

    @cachetools.cachedmethod(operator.attrgetter("_user_query_cache"), key=methodkey)
    def _get_user_features_query(self, country_column):
        return f"SELECT * FROM {self.users_features_table_name} WHERE account_id = $1 AND {country_column} = $2"

    @cachetools.cachedmethod(operator.attrgetter("_guest_user_query_cache"), key=methodkey)
    def _get_guest_user_features_query(self, country_column):
        return f"SELECT * FROM {self.users_features_table_name} WHERE account_id = -1 AND {country_column} = $1"

    @asyncache.cachedmethod(operator.attrgetter("_guest_users_cache"), key=methodkey)
    async def get_guest_user_features(
        self,
        conn,
        country_column: str,
        country_name: str
    ) -> dict:
        row: dict = await db.fetch_row_as_dict(
            conn,
            self._get_guest_user_features_query(country_column),
            country_name,
        )
        return row

    @newrelic.agent.function_trace()
    async def get_user_static_features_two_tower(
        self, customer_id: int, country_iso: str
    ) -> UserStaticFeaturesV2 | UserStaticFeaturesV3:
        country_column: str = ((self.version is TTVersion.V3) and "country_code") or "country_iso"
        async with db.connections().acquire() as conn:
            row: dict = await db.fetch_row_as_dict(
                conn,
                self._get_user_features_query(country_column),
                customer_id,
                country_iso,
            )
            if not row:
                add_transaction_attr("2towers.unknown_user", 1, SERVICE_NAME)
                LOG.warning(f"User {customer_id} not found in {country_iso}. Use default guest user features.")
                # fetch default user or get it from the cache
                row: dict = await self.get_guest_user_features(
                    conn=conn, country_column=country_column, country_name=country_iso
                )
                if not row:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail={"CODE": "DEFAULT_USER_STATIC_FEATURES_NOT_FOUND"}
                    )

        return pydantic.parse_obj_as(self.features_type, row)

    @newrelic.agent.function_trace()
    async def get_user_static_features_two_tower_df(self, customer_id: str, country_code: str) -> "pl.DataFrame":
        country_iso: str = country_code.upper()
        async with db.connections().acquire() as conn:
            return await db.fetch_as_df(
                conn,
                f"SELECT * FROM {self.users_features_table_name} WHERE account_id = $1 AND country_iso = $2",
                customer_id,
                country_iso,
            )
