import pytest

from tests.apis.common import fetch_request


@pytest.mark.parametrize("fetch_requests_paths", ["ultron/smart_query_understanding"], indirect=True)
@pytest.mark.asyncio
async def test_smart_query_understanding(ultron_client, fetch_requests_paths):
    dirs = fetch_requests_paths
    for directory in dirs:
        request, response = fetch_request(directory)

        api_response = await ultron_client.post(
            "/v1/smart-query-understanding",
            json=request,
            headers={"Content-Type": "application/json"},
        )

        assert api_response.status_code == 200
        assert api_response.headers["Content-Type"] == "application/json"
