import asyncio
from typing import TYPE_CHECKING

import aiohttp
import newrelic.agent
from aiohttp import ClientTimeout
from starlette import status

from ace.api.routes.route import get_empty_router
from ultron import SERVICE_NAME
from ultron.api.v1.items_to_purchase.models import (
    ItemsToPurchaseRequest,
    ItemToPurchaseRecommendation,
    PurchaseRecommendation,
    VendorItemMetadata,
    PurchaseItemRecommendation,
)
from ultron.config.config import get_ultron_serving_config, UltronServingConfig
from ultron.logic import rerank_recommendations, read_vendor_ids, get_global_vendor_id

if TYPE_CHECKING:
    pass

from fastapi import APIRouter, Depends


def build_router() -> APIRouter:
    router: APIRouter = get_empty_router()

    @router.post("/items-to-purchase", response_model=list[ItemToPurchaseRecommendation])
    async def items_to_purchase(
        request: ItemsToPurchaseRequest,
        vendors_ids: dict[int, str] = Depends(read_vendor_ids),
        config: UltronServingConfig = Depends(get_ultron_serving_config),
    ):
        """Endpoint that takes items to purchase in
        and returns similar items for specific vendor.
        """

        newrelic.agent.set_transaction_name(f"{SERVICE_NAME}:items-to-purchase")

        all_items = await asyncio.gather(
            *[
                process_query(
                    query,
                    vendors_ids,
                    request.vendor_id,
                    dh_qcommerce_api_endpoint=config.external_apis.dh_qcommerce_api_endpoint,
                )
                for query in request.items_to_purchase
                if query != ""
            ]
        )

        return all_items

    return router


async def process_query(query, vendors_ids, vendor_id, dh_qcommerce_api_endpoint: str):
    global_vendor_id = get_global_vendor_id(vendors_ids, vendor_id)
    similar_products = await get_grocery_items_dh(
        url=dh_qcommerce_api_endpoint, query=query, limit=5, vendor_id=global_vendor_id, country_code="ae"
    )

    extracted_items: list[PurchaseItemRecommendation] = [
        PurchaseItemRecommendation(
            item_name=item["payload"]["name"],
            global_product_id=item["payload"]["id"],
            global_vendor_id=item["payload"]["vendor_id"],
        )
        for item in similar_products["products"]["items"]
    ]

    extracted_items = await rerank_recommendations(query, extracted_items)

    similar_items_to_purchase: list[PurchaseRecommendation] = [
        PurchaseRecommendation(
            name=similar_item.item_name,
            metadata=VendorItemMetadata(
                global_vendor_id=similar_item.global_vendor_id,
                global_product_id=similar_item.global_product_id,
                product_sku="",
            ),
        )
        for similar_item in extracted_items
    ]

    return ItemToPurchaseRecommendation(original_search=query, items=similar_items_to_purchase)


async def get_grocery_items_dh(url: str, query: str, limit: int, vendor_id: str, country_code: str = "ae") -> dict:
    """
    Sends a POST request to the Delivery Hero API to retrieve groceries item data based on a query.

    Parameters:
    - query (str): The search query for products.
    - k (int): The number of products to retrieve.
    - vendor_id (str): The vendor ID to use in the query.
    - country_code (str, optional): The country code to use in the query. Defaults to "ae".

    Returns:
    - dict: A dictionary containing the API response data.
    """
    assert vendor_id, "vendor_id must be provided"
    assert limit > 0, "limit must be greater than 0"

    headers = {"Content-Type": "application/json"}

    data = {
        "filters": {"variant": 2, "category_ids": []},
        "is_darkstore": True,
        "is_migrated": True,
        "country_code": country_code,
        "query": query,
        "vendors": [{"id": vendor_id, "score": 0}],
        "brand": "talabat",
        "language_code": "en-US",
        "config": "Variation27",
        "customer_id": None,
        "limit": limit,
        "offset": 0,
        "excluded_tags": None,
    }

    async with aiohttp.ClientSession(timeout=ClientTimeout(total=3), raise_for_status=True) as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status != status.HTTP_200_OK:
                raise aiohttp.ClientResponseError(
                    response.request_info,
                    response.history,
                    status=response.status,
                    message=f"Failed to fetch {url}",
                    headers=response.headers,
                )
            data = await response.json()
    return data
