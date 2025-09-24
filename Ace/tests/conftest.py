import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pytest
import pytest_asyncio
import structlog
from _pytest.legacypath import TempdirFactory
from asgi_lifespan import LifespanManager
from httpx import AsyncClient

from abstract_ranking.two_tower import TTVersion
from ace.configs.config import StageType, AppPostgresConfig
from ace.configs.manager import ConfigManager
from ace.log import configure as configure_log
from menu_item_ranking.context import Context as MenuItemsContext, set_context as set_menu_items_context
from nba.app import build_app as build_nba_app
from ultron.service import build_app as build_ultron_app
from vendor_ranking.context import Context, set_context

if TYPE_CHECKING:
    from ace import AceService
    from structlog.stdlib import BoundLogger

LOG: "BoundLogger" = structlog.get_logger()


def clear_loggers():
    """Remove handlers from all loggers"""
    import logging

    loggers = [logging.getLogger()] + list(logging.Logger.manager.loggerDict.values())
    for logger in loggers:
        handlers = getattr(logger, "handlers", [])
        for handler in handlers:
            logger.removeHandler(handler)


@pytest.fixture(scope="session")
def event_loop():
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True, scope="session")
def setup_logging():
    configure_log(debug=True)
    yield
    clear_loggers()


@pytest.fixture(scope="session")
def postgres_config() -> AppPostgresConfig:
    return ConfigManager.load_configuration(StageType.TEST).storage.postgres


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_db(postgres_config: AppPostgresConfig, setup_logging):
    from ace.storage import db

    query_timeout = 120_000
    await db.init_connection_pool(
        service_name="tests", query_timeout=query_timeout, config=postgres_config, pool_name="TEST"
    )
    LOG.info(f"Set up DB connection with query timeout = {query_timeout}")


@pytest.fixture(scope="session")
def session_tmp_dir(tmpdir_factory: TempdirFactory) -> Path:
    yield tmpdir_factory.mktemp("s3")


@pytest_asyncio.fixture(scope="session")
async def app_context(session_tmp_dir: Path) -> Context:
    with mock.patch(
        "abstract_ranking.two_tower.artefacts_service.get_inference_path",
        new=lambda country: session_tmp_dir / f"inference_model_artifacts/{country}",
    ):
        context = await Context.instance(
            countries={
                TTVersion.V2: {"AE", "QA"},
                TTVersion.V22: {"AE", "BH", "QA"},
                TTVersion.V23: {"BH", "OM"},
                TTVersion.V3: {"BH", "OM", "KW"},
            }
        )
        async with context:
            set_context(context, overwrite=True)
            yield context


@pytest.fixture(scope="session")
def ultron_app() -> "AceService":
    app: "AceService" = build_ultron_app()
    yield app


@pytest.fixture(scope="session")
def nba_app() -> "AceService":
    app: "AceService" = build_nba_app()
    yield app


@asynccontextmanager
async def client_with_lifespan_context(app: "AceService"):
    async_client = AsyncClient(app=app.api, base_url="http://test")
    lifespan_manager = get_lifespan_manager(app)
    async with async_client, lifespan_manager:
        yield async_client


@pytest_asyncio.fixture(scope="session")
async def ultron_client(ultron_app: "AceService"):
    async with client_with_lifespan_context(ultron_app) as client:
        yield client


@pytest_asyncio.fixture(scope="session")
async def nba_client(nba_app: "AceService") -> AsyncClient:
    async with client_with_lifespan_context(nba_app) as client:
        yield client


def get_lifespan_manager(app: "AceService") -> LifespanManager:
    root_api = app._root_api  # must call lifespan of the root api application
    return LifespanManager(root_api, startup_timeout=300)


@pytest.fixture(scope="session")
def item_lifecycle_app() -> "AceService":
    from item_lifecycle.service import app

    yield app


@pytest_asyncio.fixture(scope="session")
async def item_lifecycle_client(setup_db, item_lifecycle_app: "AceService"):
    async with client_with_lifespan_context(item_lifecycle_app) as client:
        yield client


@pytest_asyncio.fixture(scope="session")
async def menu_items_app_context(session_tmp_dir: Path) -> MenuItemsContext:
    context = await MenuItemsContext.instance()
    async with context:
        set_menu_items_context(context, overwrite=True)
        yield context
