from typing import TYPE_CHECKING

import pydantic as pd
from pytest import fixture

from abstract_ranking.two_tower import TTVersion
from ace.enums import CountryShortNameUpper
from menu_item_ranking.request.input import MenuItemRequest

if TYPE_CHECKING:
    pass


@fixture(scope="function")
def menu_item_ranking_request() -> MenuItemRequest:
    raw_request = {
        "timestamp": "2023-03-03T10:03:14.5931830+00:00",
        "device_source": 4,
        "app_version": "9.1.9",
        "locale": "en-US",
        "location": {
            "country_id": 4,
            "country_code": "eg",
            "city_id": 35,
            "area_id": 1272,
            "latitude": "25.078419657090258",
            "longitude": "55.140645715858"
        },

        # use the next combination of specific values which are present in parquet file for Egypt
        "branch_id": 665933,
        "customer_id": 10914736,
    }

    return pd.parse_obj_as(MenuItemRequest, raw_request)


@fixture(scope="session")
def item_model_version() -> TTVersion:
    return TTVersion.MENUITEM_V1


@fixture(scope="session")
def item_model_country() -> CountryShortNameUpper:
    return "EG"
