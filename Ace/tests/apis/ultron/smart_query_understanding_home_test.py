from pathlib import Path
from typing import TYPE_CHECKING
import pytest

from tests.apis.common import fetch_request

if TYPE_CHECKING:
    from httpx import AsyncClient


@pytest.fixture(scope="session")
def smart_query_understanding_home_requests_path() -> Path:
    path = Path("tests/fixtures/ultron/smart_query_understanding_home")
    assert path.exists()
    return path


@pytest.mark.parametrize(
    "requests_dir_name",
    [
        "req_food",
        "req_grocery",
    ],
)
@pytest.mark.asyncio
async def test_smart_query_understanding_home(
    ultron_client: "AsyncClient", requests_dir_name: str, smart_query_understanding_home_requests_path: Path
):
    request_path = smart_query_understanding_home_requests_path / requests_dir_name
    request, expected_response = fetch_request(request_path)  # type: (dict, dict)

    api_response = await ultron_client.post(
        "/v1/smart-query-understanding-home",
        json=request,
        headers={"Content-Type": "application/json"},
    )

    assert api_response.status_code == 200
    assert api_response.headers["Content-Type"] == "application/json"
