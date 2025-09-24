import asyncio
import dataclasses
from asyncio import gather
from itertools import chain
from typing import TYPE_CHECKING, Optional, ClassVar, Tuple

import newrelic.agent
from pydantic import BaseModel, Field

from ace.storage import db
from ultron import logic
from ultron.logic import extract_comparison_sign
from ultron.runners import cross_encoder
from ultron.runners.text_embeddings import TextEmbeddingsModelName
from ultron.vector_store.data import ShoppingItemData
from ultron.vector_store.db import validate_embedding_format

if TYPE_CHECKING:
    import numpy as np
    from ultron.api.v1.items_to_purchase.models import ItemsToPurchaseRequest
    from ultron.api.v1.semantic_search.models import (
        ItemsSemanticSearchRequest,
        SemanticSearchItem,
        SortOrder,
    )


class ShoppingProduct(BaseModel):
    item_id: int
    global_vendor_id: str
    global_product_id: str
    item_name: Optional[str] = Field(alias="content")
    product_sku: Optional[int]

    class Config:
        allow_population_by_field_name = True


@dataclasses.dataclass
class VectorStore:
    table_name: ClassVar[str] = "all_items_embeddings_metadata"

    SIMILAR_PURCHASE_ITEMS_QUERY = """
                    select item_id,
                           content,
                           global_vendor_id,
                           global_product_id
                    from public.all_items_embeddings_metadata
                    where vendor_id = $1
                    order by embedding <#> $2
                    limit $3;
                """

    @newrelic.agent.function_trace()
    async def get_similar_items_to_purchase(
        self,
        request: "ItemsToPurchaseRequest",
        model_name: TextEmbeddingsModelName = TextEmbeddingsModelName.ALL_MINILM_L6_V2,
    ) -> list[list[ShoppingProduct]]:
        validate_embedding_format(table_name=self.table_name, model_name=model_name)
        embeddings: list["np.ndarray"] = await logic.get_items_embeddings(request.items_to_purchase)
        coros = [
            self._get_similar_items_to_purchase(embedding=embedding, vendor_id=request.vendor_id, limit=request.limit)
            for embedding in embeddings
        ]
        return await gather(*coros)

    @newrelic.agent.function_trace()
    async def get_similar_semantic_search_items(
        self,
        request: "ItemsSemanticSearchRequest",
        page: int,
        model_name: TextEmbeddingsModelName = TextEmbeddingsModelName.ALL_MINILM_L6_V2,
        rerank: bool = True,
    ) -> list[list[ShoppingItemData]]:
        validate_embedding_format(table_name=self.table_name, model_name=model_name)

        original_search_terms = [item.search_term for item in request.queries]
        embeddings: list["np.ndarray"] = await logic.get_items_embeddings(original_search_terms)

        # Calculate the offset
        offset = (page - 1) * request.page_size

        similar_items_tasks = [
            self._get_similar_semantic_search_items(
                request=request,
                conditions=request.queries[idx],
                embedding=embedding,
                offset=offset,
            )
            for idx, embedding in enumerate(embeddings)
        ]
        similar_items: list[list[ShoppingItemData]] = await asyncio.gather(*similar_items_tasks)

        if rerank:
            return await self._rerank_recommendations(
                requested_items=original_search_terms, recommended_items=similar_items
            )
        return similar_items

    @classmethod
    @newrelic.agent.function_trace()
    async def _rerank_recommendations(
        cls, requested_items: list[str], recommended_items: list[list[ShoppingItemData]]
    ) -> list[list[ShoppingItemData]]:
        if not any(item for item in recommended_items):
            return recommended_items

        input_pairs: list[Tuple[str, str]] = list(
            chain(
                *(
                    ((requested_item, item.content) for item in initial_recommendations)
                    for requested_item, initial_recommendations in zip(requested_items, recommended_items)
                )
            )
        )

        # Get scores for each pair
        all_scores: list[float] = await cross_encoder.get_runner().predict.async_run(input_pairs)

        groups_sizes: tuple[int] = tuple(len(value) for value in recommended_items)
        offset: int = 0
        groups_scores: list[list[float]] = [
            all_scores[offset : (offset := offset + group_size)] for group_size in groups_sizes
        ]

        # Sort the responses based on scores
        reranked_recommendations_with_score: list[list] = [
            sorted(zip(initial_recommendations, scores), key=lambda x: x[1], reverse=True)
            for initial_recommendations, scores in zip(recommended_items, groups_scores)
        ]
        return [[item[0] for item in items] for items in reranked_recommendations_with_score]

    @newrelic.agent.function_trace()
    async def _get_similar_items_to_purchase(
        self, embedding: "np.ndarray", vendor_id: int, limit: int
    ) -> list[ShoppingProduct]:
        async with db.connections().acquire() as conn:
            async with conn.transaction():
                # all embeddings must be scanned, then check all 1000 index clusters
                await conn.execute("SET LOCAl ivfflat.probes = 1000;")
                similar_items: list[ShoppingProduct] = await db.fetch_rows_as_pydantic_type(
                    conn,
                    self.SIMILAR_PURCHASE_ITEMS_QUERY,
                    ShoppingProduct,
                    vendor_id,
                    embedding,
                    limit,
                )
        return similar_items

    @newrelic.agent.function_trace()
    async def _get_similar_semantic_search_items(
        self,
        request: "ItemsSemanticSearchRequest",
        conditions: "SemanticSearchItem",
        embedding: "np.ndarray",
        offset: int,
    ) -> list[ShoppingItemData]:
        ITEMS_SEMANTIC_SEARCH_QUERY = """select * from f_items_semantic_search($1,$2,$3,$4, $5, $6, $7, $8, $9, $10)"""

        if conditions.sort_by:
            # 1) rename key names
            # 2) remove none values

            def _rename_key(_key: str) -> str:
                return (
                    _key.replace("price", "avg_original_item_price_lc")
                    .replace("number_of_orders", "order_count")
                    .strip()
                )

            sort_by: dict[str, SortOrder] = {
                _rename_key(key): sort_order for key, sort_order in conditions.sort_by.items() if sort_order is not None
            }

            order_conditions = ", ".join(f"{key} {sort_order.value}" for key, sort_order in sort_by.items())
            ITEMS_SEMANTIC_SEARCH_QUERY += f" ORDER BY {order_conditions}"

        if filter_by := conditions.filter_by:
            price_condition: str = filter_by.get("price")
            comparison_sign, price = extract_comparison_sign(price_condition)
        else:
            comparison_sign, price = "", 0.0

        async with db.connections().acquire() as conn:
            async with conn.transaction():
                await conn.execute("SET LOCAl ivfflat.probes = 35;")
                similar_items = await db.fetch_rows_as_pydantic_type(
                    conn,
                    ITEMS_SEMANTIC_SEARCH_QUERY,
                    ShoppingItemData,
                    request.country_code,
                    [item.value for item in conditions.verticals],
                    request.chains,
                    embedding,
                    request.page_size,
                    offset,
                    comparison_sign,
                    price,
                    conditions.past_orders_only,
                    request.customer_id,
                )
                return similar_items
