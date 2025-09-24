import asyncio
import datetime as dt
from random import randint
from typing import TYPE_CHECKING, Type

import fastapi
import newrelic.agent
import structlog

import ace
from ace.configs.config import AppConfig, AppPostgresConfig
from ace.configs.manager import ConfigManager, EnvAppSettings
from ace.model_log import LogEntry, log_execution
from ace.storage import db
from nba import SERVICE_NAME, data, hero_banner
from nba.config.config import get_nba_serving_config
from nba.hero_banner import agents
from nba.input import CountryCode, HeroBannersRequest, HeroBannersResponse

if TYPE_CHECKING:
    from nba.base_logic import BaseLogic
    from structlog.stdlib import BoundLogger

APP_CONFIG: AppConfig = ConfigManager.load_configuration(stage=EnvAppSettings().stage)

AGENTS_MANAGER = agents.AgentsManager(
    base_dir=agents.get_inference_path(),
    s3_app_config=APP_CONFIG.storage.s3,
)

LOG: "BoundLogger" = structlog.get_logger()
BACKGROUND_TASK_RANDOM_WAIT_SEC: int = 10
ARTIFACTS_LATEST_DATE: dt.datetime | None = None


async def random_wait(max_wait_sec: int):
    wait_sec = randint(0, max_wait_sec)  # [0, max_wait_sec]
    await asyncio.sleep(wait_sec)


def check_models_and_banners_date(models_date: dt.date, banners_date: dt.date):
    """In the best case, models' date should be today, banners' date should be yesterday.
    If that condition is not true, then just log error and continue working.
    """
    dates_are_compatible = True
    msg = "Validating models and banners compatibility. "

    # Check models' date
    today = AGENTS_MANAGER.day_with_offset(date_offset=0)
    if models_date != today:
        msg += f"Models are NOT LATEST: actual date({models_date}) should be today({today}). "
        dates_are_compatible = False
    else:
        msg += f"Models are UP TO DATE {today=}. "

    # Check banners' date
    yesterday = AGENTS_MANAGER.day_with_offset(date_offset=1)
    if banners_date != yesterday:
        msg += f"Banners are NOT LATEST: actual date({banners_date}) should be yesterday({yesterday}). "
        dates_are_compatible = False
    else:
        msg += f"Banner are UP TO DATE yesterday({yesterday}). "

    if dates_are_compatible:
        LOG.info(msg)
    else:
        LOG.error(msg)


@newrelic.agent.function_trace()
async def refresh_banners_data() -> dt.date:
    await random_wait(max_wait_sec=BACKGROUND_TASK_RANDOM_WAIT_SEC)
    _, banners_latest_snapshot_date = await data.refresh_banners_inference_data()
    return banners_latest_snapshot_date


@newrelic.agent.function_trace()
async def refresh_models() -> dt.date:
    global ARTIFACTS_LATEST_DATE
    LOG.info("Refreshing models STARTED", debug=ace.DEBUG, stage=ace.STAGE)
    await random_wait(max_wait_sec=BACKGROUND_TASK_RANDOM_WAIT_SEC)
    new_artifacts_loaded, date_offset = await AGENTS_MANAGER.download_model_artifacts(date_offset=0)
    if new_artifacts_loaded:
        ARTIFACTS_LATEST_DATE = AGENTS_MANAGER.day_with_offset(date_offset)
        AGENTS_MANAGER.reload_models()
    LOG.info(
        f"Refreshing models FINISHED: "
        f"models reloaded: {new_artifacts_loaded}, new models date: {ARTIFACTS_LATEST_DATE or 'UNKNOWN'}",
        debug=ace.DEBUG,
        stage=ace.STAGE,
    )

    return ARTIFACTS_LATEST_DATE


def build_app():
    app = ace.AceService(SERVICE_NAME, runners=[])
    postgres_config: "AppPostgresConfig" = get_nba_serving_config().storage.postgres

    @app.on_background_startup()
    async def setup_background_context():
        await db.init_connection_pool(  # DB connection pool (for background tasks)
            service_name=SERVICE_NAME,
            pool_name="background",
            min_size=0,
            max_size=5,
            query_timeout=postgres_config.background_query_timeout,
            config=postgres_config,
        )
        LOG.info("Background context is prepared", debug=ace.DEBUG, stage=ace.STAGE)

    @app.on_api_startup()
    async def setup_db():  # DB connection pool (API thread)
        await db.init_connection_pool(
            service_name=SERVICE_NAME,
            pool_name="main",
            query_timeout=postgres_config.main_query_timeout,
            config=postgres_config,
        )

    @app.on_api_shutdown()
    async def stop():
        await db.clear_connection_pool()

    @app.background.task("every 55 minutes")
    @newrelic.agent.background_task(name=f"{SERVICE_NAME}:bg_refresh_model_and_banners_data")
    async def refresh_data():
        banners_latest_snapshot_date, artifacts_latest_date = await asyncio.gather(
            refresh_banners_data(), refresh_models()
        )  # type: (dt.date, dt.date)
        check_models_and_banners_date(models_date=artifacts_latest_date, banners_date=banners_latest_snapshot_date)

    @app.ready.check()
    def is_ready(_) -> bool:
        return data.is_loaded()

    @app.api.get("/home/v1/{country_code}/customer/{customer_id}/hero-banners")
    async def handler(
        country_code: CountryCode,
        customer_id: int,
        background_tasks: fastapi.BackgroundTasks,
    ) -> HeroBannersResponse:
        logic: Type["BaseLogic"] = hero_banner.Logic
        newrelic.agent.set_transaction_name(logic.NAME)

        request = HeroBannersRequest(country_code=country_code, customer_id=customer_id)
        execution_log = LogEntry(
            SERVICE_NAME,
            {
                "country_code": country_code.value,
                "customer_id": customer_id,
            },
        )
        execution = logic(
            request=request,
            agents_manager=AGENTS_MANAGER,
            exec_log=execution_log,
        )

        await execution.prepare_features()
        final_sorting = await execution.predict()

        execution_log.response = final_sorting.dict()  # TODO Add each model's response to the log
        background_tasks.add_task(log_execution, execution_log)

        return final_sorting

    return app
