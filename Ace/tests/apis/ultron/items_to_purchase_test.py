from pathlib import Path
from typing import TYPE_CHECKING

import pydantic
import pytest
from starlette import status

from tests.apis.common import fetch_request

from ultron.api.v1.items_to_purchase.models import ItemToPurchaseRecommendation

if TYPE_CHECKING:
    from httpx import AsyncClient, Response


@pytest.fixture(scope="session")
def items_to_purchase_requests_path() -> Path:
    path = Path("tests/fixtures/ultron/items_to_purchase")
    assert path.exists()
    return path


@pytest.mark.parametrize(
    "requests_dir_name",
    [
        "req_food_vendor",
        "req_groceries_vendor",
    ],
)
@pytest.mark.asyncio
async def test_items_to_purchase(
    ultron_client: "AsyncClient", requests_dir_name: str, items_to_purchase_requests_path: Path
):
    request_path = items_to_purchase_requests_path / requests_dir_name
    request, expected_response = fetch_request(request_path)  # type: (dict, dict)

    api_response: "Response" = await ultron_client.post(
        "/v1/items-to-purchase",
        json=request,
        headers={"Content-Type": "application/json"},
    )

    assert api_response.status_code == status.HTTP_200_OK
    assert api_response.headers["Content-Type"] == "application/json"

    response: list[dict] = pydantic.parse_obj_as(list[dict], api_response.json())

    assert len(response) == len(expected_response)

    for response_item in response:
        pydantic.parse_obj_as(ItemToPurchaseRecommendation, response_item)
