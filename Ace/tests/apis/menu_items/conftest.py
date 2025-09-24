import pytest
import pytest_asyncio

from ace import AceService
from menu_item_ranking.app import build_menu_items_ranking_app
from menu_item_ranking.context import Context
from tests.conftest import client_with_lifespan_context


@pytest.fixture(scope="session")
def menu_items_app() -> "AceService":
    app: "AceService" = build_menu_items_ranking_app()
    yield app


@pytest_asyncio.fixture(scope="session")
async def menu_items_client(menu_items_app_context: Context, menu_items_app: "AceService"):
    async with client_with_lifespan_context(menu_items_app) as client:
        yield client
