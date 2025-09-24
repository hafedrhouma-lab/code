import json
from pathlib import Path
from pprint import pformat
from unittest import mock

import pytest
from httpx import AsyncClient
from starlette import status

from tests.apis.common import fetch_request


@pytest.fixture
def personalized_ranking_fetch_requests_path() -> Path:
    return Path("tests/fixtures/vendor_ranking/personalized_ranking")


@pytest.mark.parametrize(
    "test_dir_name", [
        "base",
        "no_fast_sort_with_penalty"
    ]
)
@pytest.mark.asyncio
async def test_personalized_ranking(
    vendor_ranking_client: AsyncClient, test_dir_name: str, personalized_ranking_fetch_requests_path: Path
):
    request, expected_response = fetch_request(personalized_ranking_fetch_requests_path / test_dir_name)

    api_response = await vendor_ranking_client.post("/sort", json=request, headers={"Content-Type": "application/json"})
    response = json.load(api_response)

    assert api_response.status_code == status.HTTP_200_OK
    assert response == expected_response, pformat(response)


@pytest.mark.parametrize("test_dir_name", ["fast_sort"])
@pytest.mark.asyncio
async def test_personalized_rank_fast_sort(
    vendor_ranking_client: AsyncClient, test_dir_name: str, personalized_ranking_fetch_requests_path: Path
):
    with mock.patch("vendor_ranking.personalized_ranking.logic.Logic.TOP_N_FAST_SORT_LIMIT", new=20):
        request, expected_response = fetch_request(personalized_ranking_fetch_requests_path / test_dir_name)

        api_response = await vendor_ranking_client.post(
            "/sort", json=request, headers={"Content-Type": "application/json"}
        )

        response, status_code = json.load(api_response), api_response.status_code
        assert status_code == status.HTTP_200_OK
        assert response == expected_response
