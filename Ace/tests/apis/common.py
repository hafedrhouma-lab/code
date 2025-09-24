import json
from pathlib import Path
from typing import Tuple


def fetch_request(directory_path) -> Tuple[dict, dict]:
    if isinstance(directory_path, Path):
        assert directory_path.exists()

    request_path = f"{directory_path}/request.json"
    response_path = f"{directory_path}/response.json"

    try:
        with open(f"{request_path}") as request:
            with open(f"{response_path}") as response:
                request_json = json.load(request)
                response_json = json.load(response)
    except FileNotFoundError as ex:
        raise FileNotFoundError(f"File not found: {request_path} or {response_path}") from ex

    return request_json, response_json
