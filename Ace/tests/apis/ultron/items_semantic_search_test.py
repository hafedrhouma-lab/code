import math
from pathlib import Path

import pydantic
import pytest
from httpx import AsyncClient

from tests.apis.common import fetch_request


@pytest.fixture(scope="session")
def semantic_search_requests_path() -> Path:
    return Path("tests/fixtures/ultron/items_semantic_search")


@pytest.mark.parametrize(
    "requests_dir_name",
    [
        "fish_in_grocery",
    ],
)
@pytest.mark.asyncio
async def test_items_semantic_search(
    ultron_client: "AsyncClient", requests_dir_name: str, semantic_search_requests_path: Path
):
    request, expected_response = fetch_request(semantic_search_requests_path / requests_dir_name)

    api_response = await ultron_client.post(
        "/v1/items-semantic-search",
        json=request,
        headers={"Content-Type": "application/json"},
    )

    response: list[dict] = pydantic.parse_raw_as(list[dict], api_response.text)  # parse and validate
    assert len(response) == len(expected_response)
    for response_item, expected_item in zip(response, expected_response):
        assert response_item["original_search"] == expected_item["original_search"]

        similar_items, expected_similar_items = response_item["similar_items"], expected_item["similar_items"]
        for similar_item, expected_similar_item in zip(similar_items, expected_similar_items):
            distance, expected_distance = (similar_item.pop("distance"), expected_similar_item.pop("distance"))
            assert similar_item == expected_similar_item
            assert math.isclose(distance, expected_distance, rel_tol=0.00001)
