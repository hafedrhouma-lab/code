import logging

import pydantic as pd
from asyncache import cached
from cachetools import LRUCache
from fastapi import Request, HTTPException
from pydantic import BaseModel
from starlette import status

from ace.newrelic import add_transaction_attr
from ace.storage import db
from menu_item_ranking import SERVICE_NAME
from menu_item_ranking.request.input import MenuItemRequest

LOG = logging.getLogger(__name__)


QUERY = """
    SELECT chain_id, vendor_id, country_code 
    FROM vendor_to_chain_mapping_two_tower
    WHERE vendor_id = $1 AND country_code = $2
"""


class ChainMappingData(BaseModel):
    chain_id: int
    vendor_id: int
    country_code: str


@cached(cache=LRUCache(maxsize=1024))
async def get_chain_for_vendor(vendor_id: int, country_code: str) -> int:
    async with db.connections().acquire() as conn:
        rows: list[ChainMappingData] | None = await db.fetch_rows_as_pydantic_type(
            conn, QUERY, ChainMappingData, vendor_id, country_code
        )
        if not rows:
            add_transaction_attr("unknown_chain", 1, SERVICE_NAME)
            raise HTTPException(
                status_code=status.HTTP_406_NOT_ACCEPTABLE,
                detail={
                    "code": "CHAIN_ID_NOT_FOUND",
                    "description": f"chain_id is not found for (vendor/branch, country).",
                },
            )
        if len(rows) != 1:
            raise HTTPException(
                status_code=status.HTTP_406_NOT_ACCEPTABLE,
                detail={
                    "code": "INCONSISTENT_CHAIN_ID",
                    "description": f"there are multiple chain_id for (vendor/branch, country).",
                },
            )
        chain_id = rows[0].chain_id
        LOG.debug(f"Found chain_id({chain_id}) for vendor({vendor_id}) and county({country_code})")
        return chain_id


async def get_request_with_chain_id(request: Request) -> MenuItemRequest:
    payload: MenuItemRequest = pd.parse_raw_as(MenuItemRequest, await request.body())
    if payload.chain_id is None:
        payload.chain_id = await get_chain_for_vendor(
            vendor_id=payload.vendor_id, country_code=payload.location.country_code
        )
    return payload

