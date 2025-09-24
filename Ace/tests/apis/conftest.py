import os
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from ace.storage import db
from tests.conftest import client_with_lifespan_context
from vendor_ranking import data
from vendor_ranking.context import Context

if TYPE_CHECKING:
    from ace import AceService


@pytest.fixture(scope="session")
def nba_app() -> "AceService":
    from nba.service import app

    yield app


@pytest_asyncio.fixture(scope="session")
async def nba_client(app_context: "Context", nba_app: "AceService"):
    async with client_with_lifespan_context(nba_app) as client:
        await db.init_connection_pool()
        yield client


@pytest.fixture(scope="session")
def vendor_ranking_app() -> "AceService":
    from vendor_ranking.service import app

    yield app


@pytest_asyncio.fixture(scope="session")
async def vendor_ranking_client(app_context: "Context", vendor_ranking_app: "AceService"):
    async with client_with_lifespan_context(vendor_ranking_app) as client:
        await db.init_connection_pool(query_timeout=500)

        await data.refresh_vendors()
        await data.refresh_ranking_penalties()

        yield client


@pytest.fixture
def fetch_requests_paths(request):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", f"fixtures/{request.param}")
    dir_list = os.listdir(path)

    files_to_be_removed = ["__init__.py"]
    return [os.path.join(path, file) for file in dir_list if file not in files_to_be_removed]
