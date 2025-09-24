from operator import itemgetter, attrgetter
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ace.storage import db
from ultron import logic
from ultron.api.v1.items_to_purchase.models import ItemsToPurchaseRequest
from ultron.runners.text_embeddings import TextEmbeddingsModelName
from ultron.vector_store.data import ShoppingItemData
from ultron.vector_store.store import VectorStore, ShoppingProduct

if TYPE_CHECKING:
    pass


async def get_all_items_with_embeddings(vendor_id: int) -> list[dict]:
    query = """
        select item_id,
               global_vendor_id,
               global_product_id,
               embedding
        from public.all_items_embeddings_metadata
        where vendor_id = $1;
    """

    async with db.connections().acquire() as conn:
        items: list[dict] = await db.fetch_rows_as_dicts(conn, query, vendor_id)
    return items


@pytest.fixture(scope="function")
def vector_store() -> VectorStore:
    return VectorStore()


def create_shopping_item(name: str) -> ShoppingItemData:
    return ShoppingItemData(
        content=name,
        item_id=0,
        chain_id=0,
        chain_name="",
        order_count=0,
        unique_order_dates=0,
        vertical="",
        distance=0.0,
    )


@pytest.mark.usefixtures("ultron_client")
class VectorStoreTest:
    @pytest.mark.asyncio
    async def test_similar_items_to_purchase_search(self):
        vendor_id = 44692
        item_to_purchase = "fish"

        # Step 1: request PGVector with ranking by dot product
        request = ItemsToPurchaseRequest(vendor_id=vendor_id, items_to_purchase=[item_to_purchase], limit=5)
        vector_store = VectorStore()
        similar_items_result: list[list["ShoppingProduct"]] | None = await vector_store.get_similar_items_to_purchase(
            request=request, model_name=TextEmbeddingsModelName.ALL_MINILM_L6_V2
        )
        assert similar_items_result and len(similar_items_result) == 1
        similar_items = similar_items_result[0]

        # Step 2: calculate dot products manually
        embeddings: list["np.ndarray"] = await logic.get_items_embeddings(request.items_to_purchase)
        embedding: "np.ndarray" = embeddings[0]

        items: list[dict] = await get_all_items_with_embeddings(vendor_id=vendor_id)
        for item in items:
            item["rank"] = float(np.dot(item["embedding"], embedding))

        # Step 3: check results
        items.sort(key=itemgetter("rank"), reverse=True)
        expected_similar_items = items[: request.limit]
        expected_items_ids = list(map(itemgetter("item_id"), expected_similar_items))
        items_ids = list(map(attrgetter("item_id"), similar_items))
        assert expected_items_ids == items_ids

    @pytest.mark.asyncio
    async def test_rerank_similar_items_to_purchase(self, vector_store: VectorStore):
        requested_items: list[str] = ["fish"]
        recommended_items_groups: list[list[ShoppingItemData]] = [
            [
                create_shopping_item(name)
                for name in (
                    "Oceano Fresh Salmon Portion 200 g Category: Meat & Fish Sub category: Fish & Seafood*",
                    "Oceano Fresh Tuna Loin Portion 300 g Category: Meat & Fish Sub category: Fish & Seafood*",
                    "Oceano Fresh Salmon Portion 300 g Category: Meat & Fish Sub category: Fish & Seafood*",
                    "Oceano Fresh Salmon Steak 200 g Category: Meat & Fish Sub category: Fish & Seafood*",
                    "Oceano Fresh Organic Salmon Portion 300 g Category: Meat & Fish Sub category: Fish & Seafood*",
                )
            ]
        ]

        reranked_items_groups: list[list[ShoppingItemData]] = await vector_store._rerank_recommendations(
            requested_items=requested_items, recommended_items=recommended_items_groups
        )
        assert len(reranked_items_groups) == len(recommended_items_groups)
        for reranked_items, recommended_items in zip(reranked_items_groups, recommended_items_groups):
            reranked_names = [item.content for item in reranked_items]
            recommended_names = [item.content for item in recommended_items]
            assert reranked_names != recommended_names
            assert sorted(reranked_names) == sorted(recommended_names)
