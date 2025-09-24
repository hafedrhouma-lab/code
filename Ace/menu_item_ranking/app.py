from typing import TYPE_CHECKING, Type, Optional

import fastapi
import newrelic.agent
import structlog
from fastapi import Request, Depends
from rocketry.conditions.api import time_of_day, time_of_week
from rocketry.conds import weekly, hourly
from structlog.stdlib import BoundLogger

import ace
from abstract_ranking.background_tasks import random_wait
from abstract_ranking.config import LogicsConfig
from abstract_ranking.input import FunWithFlags
from ace.model_log import LogEntry, log_execution
from ace.newrelic import add_transaction_attrs, add_transaction_attr
from ace.storage import db
from menu_item_ranking import SERVICE_NAME
from menu_item_ranking.configs.config import (
    get_menu_item_ranking_config,
    MenuItemRankingLogicConfig,
)
from menu_item_ranking.context import Context as MenuItemRankingContext
from menu_item_ranking.context import set_context, ctx
from menu_item_ranking.logic.base_logic import MenuItemBaseLogic, MenuItemLogic
from menu_item_ranking.request.chain_id_provider import get_request_with_chain_id
from menu_item_ranking.request.input import MenuItemRequest
from menu_item_ranking.settings import (
    ExperimentSettings,
    LogicType,
    get_logics_registry
)

if TYPE_CHECKING:
    from abstract_ranking.base_logic import BaseLogic


LOG: "BoundLogger" = structlog.get_logger()
BACKGROUND_TASK_RANDOM_WAIT_SEC: int = 30


def _get_logic(fwf: FunWithFlags, experiment: ExperimentSettings) -> Optional[LogicType]:
    """ Check if there is a logic for experiment specified in FWF."""
    if active_experiment := fwf.active_experiments.get(experiment.name):
        # Without the default value getattr() will raise AttributeError if there is no such attribute
        if logic := getattr(experiment.logic, active_experiment.variation, None):
            add_transaction_attr("fwf.experiment", experiment.name, SERVICE_NAME)
            LOG.debug(
                f"Selected logic name={logic.NAME}, "
                f"nickname={logic.NICKNAME or ''}, "
                f"experiment={experiment.name}, "
                f"variation={active_experiment.variation}"
            )
            return logic


def _select_logic(country_code: str, config: LogicsConfig, logics_registry: dict[str, Type["MenuItemBaseLogic"]]):
    default_logic = logics_registry[config.default]
    for logic_nickname, countries in config.countries_logic.items():
        if country_code in countries:
            if logic := logics_registry.get(logic_nickname):
                return logic
            # We can just count logs here (instead of writing custom metric), as this is not something "usual"
            LOG.warning(f"No logic found for {logic_nickname} in {country_code}. Use default.")
            return default_logic

    return default_logic


def select_default_logic(country_code: str, config: MenuItemRankingLogicConfig,
                         logics_registry: dict[str, Type["MenuItemBaseLogic"]]):
    return _select_logic(country_code, config=config.logic.default, logics_registry=logics_registry)


def select_control_logic(country_code: str, config: MenuItemRankingLogicConfig,
                         logics_registry: dict[str, Type["MenuItemBaseLogic"]]):
    return _select_logic(country_code, config=config.logic.control, logics_registry=logics_registry)


def validate_fwf(fwf: FunWithFlags, country_code: str, customer_id: int = 0):
    user_experiments = [
        (experiment_name, experiment)
        for experiment_name, experiment in fwf.active_experiments.items()
        if experiment.variation == "Variation1"
    ]
    if len(user_experiments) > 1:
        # We can just count logs here for alerts, as this is not something "usual"
        LOG.debug(
            f"More than one experiment detected for user:"
            f" country={country_code}, customer_id={customer_id}: {list(user_experiments)}"
        )
        add_transaction_attr("more_than_one_experiment", 1, SERVICE_NAME)


def select_logic(
    fwf: FunWithFlags,
    country_code: str,
    config: MenuItemRankingLogicConfig,
    logics_registry: dict[str, Type["MenuItemBaseLogic"]],
    customer_id: Optional[int] = 0,
) -> Type["MenuItemBaseLogic"]:
    return MenuItemLogic


async def check_all_execution_requirements(msg: str):
    # TODO: implement `check_execution_requirements()` for all other experiments
    return


def build_menu_items_ranking_app() -> ace.AceService:
    app = ace.AceService(
        name=SERVICE_NAME,
        runners=[],
    )

    @app.on_background_startup()
    async def setup_context():
        context = await MenuItemRankingContext.instance()
        pg_config = context.app_config.storage.postgres
        await db.init_connection_pool(  # DB connection pool (for background tasks)
            service_name=SERVICE_NAME,
            pool_name="background",
            min_size=0,
            max_size=5,
            query_timeout=pg_config.background_query_timeout,
            config=pg_config,
        )
        await check_all_execution_requirements(msg="background")
        await context.open()
        set_context(context)
        LOG.info("Background tasks preparation DONE", debug=ace.DEBUG, stage=ace.STAGE)

    @app.on_api_startup()
    async def setup_db():  # DB connection pool (API thread)
        config = get_menu_item_ranking_config().storage.postgres
        await db.init_connection_pool(
            pool_name="main",
            service_name=SERVICE_NAME,
            config=config,
            query_timeout=config.main_query_timeout
        )
        await check_all_execution_requirements(msg="main")

    @app.on_api_shutdown()
    async def stop():
        if context := ctx():
            await context.close()
        await db.clear_connection_pool()

    @app.ready.check()
    def is_ready(_) -> bool:
        """ K8s readiness criteria:
            1) Context created and opened: user models are loaded
            2) Database's data is loaded into memory
        """
        if (context := ctx()) and context.opened:
            return True

        # TODO readiness check: `return data.is_loaded()`
        return False

    @app.background.task(hourly)
    @newrelic.agent.background_task(name=f"{SERVICE_NAME}:refresh_user_models")
    async def refresh_user_models() -> None:
        if (context := ctx()) and context.opened:
            await random_wait(max_wait_sec=BACKGROUND_TASK_RANDOM_WAIT_SEC)
            await context.reload_tt_user_models()
        else:
            LOG.info("App context in not ready to reload user model")

    @app.background.task(
        weekly & time_of_week.at("Mon") & time_of_day.between("00:00", "00:01")
    )
    @newrelic.agent.background_task(name=f"{SERVICE_NAME}:refresh_items_embeddings")
    async def refresh_items_embeddings() -> None:
        if (context := ctx()) and context.opened:
            await random_wait(max_wait_sec=BACKGROUND_TASK_RANDOM_WAIT_SEC)
            await context.reload_items_embeddings()
        else:
            LOG.info("App context in not ready to reload items embeddings")

    @app.api.post("/v1/sort")
    @app.api.post("/sort")
    async def handler(
        http_request: Request,
        background_tasks: fastapi.BackgroundTasks,
        request: MenuItemRequest = Depends(get_request_with_chain_id),
        context: MenuItemRankingContext = Depends(ctx),
        config: MenuItemRankingLogicConfig = Depends(get_menu_item_ranking_config),
        logics_registry: dict[str, Type["MenuItemBaseLogic"]] = Depends(get_logics_registry),
    ) -> list[int]:
        logic: Type["MenuItemBaseLogic"] = select_logic(
            fwf=request.fwf,
            country_code=request.location.country_code,
            config=config,
            logics_registry=logics_registry,
            customer_id=request.customer_id,
        )
        execution_log = LogEntry(SERVICE_NAME, await http_request.json())

        execution: "BaseLogic" = logic(request=request, exec_log=execution_log, context=context)
        newrelic.agent.set_transaction_name(execution.NAME)
        add_transaction_attrs(
            (
                # Can be extracted from the transaction name, but better to have it as a separate attribute also
                ("logic", MenuItemBaseLogic.NAME),
                ("country_code", request.location.country_code.upper()),
                ("country", request.location.country_code.upper()),  # Backward compatibility with Two Towers
                ("customer_id", request.customer_id),
                ("model_tag", logic.MODEL_TAG),
                ("model_version", logic.VERSION),
            ),
            service_name=SERVICE_NAME,
        )

        await execution.prepare_features()
        execution_log.response = final_sorting = await execution.sort()

        execution.push_transaction_stats()
        background_tasks.add_task(log_execution, execution_log)

        return final_sorting

    return app
