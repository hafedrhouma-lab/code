import json
from pathlib import Path
from unittest import mock

import pytest

from tests.apis.common import fetch_request


@pytest.fixture(
    params=[directory for directory in Path("tests/fixtures/ultron/recipe_stream").iterdir() if directory.is_dir()]
)
def ultron_request_response(request) -> (dict, dict):
    return fetch_request(request.param)


@pytest.mark.asyncio
async def test_recipe_stream(ultron_client, ultron_request_response: (dict, dict)):
    request_data, expected_response = ultron_request_response

    # try to stream in bigger chunks
    with mock.patch("ultron.logic.Conversation.number_of_words", new=50):
        api_response = await ultron_client.post(
            "/v1/recipe-chatbot",
            json=request_data,
            headers={"Content-Type": "application/json"},
        )

    assert api_response.status_code == 200
    if request_data["stream"]:
        assert api_response.headers["Content-Type"] == "text/event-stream; charset=utf-8"
    else:
        assert api_response.headers["Content-Type"] == "application/json"
        json_response = json.loads(api_response.content)
        assert json_response["role"] == "assistant"
