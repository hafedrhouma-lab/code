from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from starlette import status

from tests.apis.common import fetch_request

if TYPE_CHECKING:
    from httpx import AsyncClient


@pytest.mark.parametrize(
    "test_dir_name",
    ("valid_request", "customer_exist_with_no_categories", "customer_does_not_exist"),
)
@pytest.mark.asyncio
async def test_item_lifecycle_responses(item_lifecycle_client: "AsyncClient", test_dir_name: str):
    fetch_requests_path = Path(f"tests/fixtures/item_lifecycle/{test_dir_name}")
    request, expected_response = fetch_request(fetch_requests_path)  # type: (dict, dict)
    customer_id = request.get("customer_id")
    country_code = request.get("country_code")

    api_response = await item_lifecycle_client.get(f"/home/v1/{country_code}/customer/{customer_id}/item_replenishment")

    assert api_response.status_code == status.HTTP_200_OK
    response: dict = api_response.json()
    assert response == expected_response
