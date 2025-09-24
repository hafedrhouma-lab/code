import json
from pathlib import Path
from pprint import pformat

import pytest
from httpx import AsyncClient
from starlette import status

from tests.apis.common import fetch_request


@pytest.fixture()
def price_parity_ranking_fetch_requests_path() -> Path:
    return Path("tests/fixtures/vendor_ranking/price_parity_ranking")


@pytest.mark.parametrize(
    "test_dir_name", [
        "req1",
        "twotowers_and_penalization",
    ]
)
@pytest.mark.asyncio
async def test_price_parity_rank(
    vendor_ranking_client: AsyncClient,
    test_dir_name: str,
    price_parity_ranking_fetch_requests_path: Path
):
    request, expected_response = fetch_request(price_parity_ranking_fetch_requests_path / test_dir_name)

    api_response = await vendor_ranking_client.post(
        "/sort", json=request, headers={"Content-Type": "application/json"}
    )
    response = json.load(api_response)
    response_code = api_response.status_code

    assert response_code == status.HTTP_200_OK, f"[{test_dir_name}], wrong response code {response_code}: {response}"
    assert response == expected_response, f"[{test_dir_name}], wrong response payload, {pformat(response)}"
