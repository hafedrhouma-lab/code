import pydantic
import pytest
from httpx import AsyncClient

from abstract_ranking.two_tower import TTVersion
from abstract_ranking.two_tower.artefacts_service import CountryServingConfig
from ace.enums import CountryShortNameUpper


@pytest.fixture(scope="session")
def customer_branch_per_config() -> dict[(CountryShortNameUpper, int), (int, int)]:
    return {
        (country, recall): (customer_id, branch_id, chain_id)
        for country, recall, customer_id, branch_id, chain_id in (
            ("EG", 3765, 10914736, 710047, 665933),
            ("EG", 3766, 28066673, 518431, 506073),
        )
    }


@pytest.fixture(scope="session")
def country_serving_config() -> CountryServingConfig:
    return CountryServingConfig(
        country="EG",
        recall=3765,
        version=TTVersion.MENUITEM_V1
    )


@pytest.fixture(scope="session")
def menu_items_request(
    customer_branch_per_config: dict[(CountryShortNameUpper, int), (int, int, int)],
    country_serving_config: CountryServingConfig
) -> dict:
    customer_id, branch_id, chain_id = customer_branch_per_config[
        (country_serving_config.country, country_serving_config.recall)
    ]
    return {
        "timestamp": "2023-03-03T10:03:14.5931830+00:00",
        "device_source": 4,
        "app_version": "9.1.9",
        "locale": "en-US",
        "location": {
            "country_id": 4,
            "country_code": country_serving_config.country,
            "city_id": 35,
            "area_id": 1272,
            "latitude": "25.078419657090258",
            "longitude": "55.140645715858"
        },
        # use the next combination of specific values which are present in parquet file for Egypt
        "branch_id": branch_id,
        "chain_id": chain_id,
        "customer_id": customer_id,
    }


@pytest.mark.asyncio
async def test_menu_items_ranking(
    menu_items_client: "AsyncClient", menu_items_request: dict
):
    api_response = await menu_items_client.post(
        "/v1/sort",
        json=menu_items_request,
        headers={"Content-Type": "application/json"},
    )

    response: list[int] = pydantic.parse_raw_as(list[int], api_response.text)  # parse and validate
    assert response is not None
    assert len(response) > 0
