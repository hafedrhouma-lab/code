from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from starlette import status

from tests.apis.common import fetch_request

if TYPE_CHECKING:
    from httpx import AsyncClient


@pytest.mark.parametrize(
    "test_dir_name",
    (
        "ae/base",
        "ae/one_absent_chain",
        "ae/all_absent_chains",
        "qa/as_wrong_country",
    ),
)
@pytest.mark.asyncio
async def test_two_towers_v2_rank(vendor_ranking_client: "AsyncClient", test_dir_name: str):
    fetch_requests_path = Path(f"tests/fixtures/vendor_ranking/two_towers_ranking/api/{test_dir_name}")
    request, expected_response = fetch_request(fetch_requests_path)  # type: (dict, dict)

    api_response = await vendor_ranking_client.post("/sort", json=request, headers={"Content-Type": "application/json"})

    assert api_response.status_code == status.HTTP_200_OK
    response: dict = api_response.json()
    assert response == expected_response
