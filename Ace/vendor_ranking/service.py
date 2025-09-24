#!/usr/bin/env python3
import asyncio
from typing import TYPE_CHECKING, Type, Optional

import fastapi
import newrelic
import newrelic.agent
import structlog
from fastapi import Request, Depends

import ace
from abstract_ranking.background_tasks import random_wait
from abstract_ranking.config import LogicsConfig
from abstract_ranking.input import FunWithFlags
from ace.model_log import LogEntry, log_execution
from ace.newrelic import add_transaction_attrs, add_transaction_attr
from ace.storage import db
from vendor_ranking import SERVICE_NAME, personalized_ranking, data
from vendor_ranking.configs.config import (
    get_vendor_ranking_config,
    VendorRankingConfig,
    RankingLogicConfig,
)
from vendor_ranking.context import Context, set_context, ctx
from vendor_ranking.input import VendorList
from vendor_ranking.settings import (
    PRICE_PARITY_EXPERIMENT,
    PRICE_PARITY_V2_EXPERIMENT,
    TWO_TOWERS_V2_EXPERIMENT,
    KPP_TWO_TOWERS_V2_EXPERIMENT,
    CATBOOST_NO_FAST_SORT_EXPERIMENT,
    ExperimentSettings,
    get_logics_registry,
    PERSONALIZED_RANKING_TT_V3, VARIATION1,
)
from vendor_ranking.two_tower.logic import LogicTwoTowersV23, LogicTwoTowersV3

if TYPE_CHECKING:
    from vendor_ranking.settings import VendorLogicType
    from structlog.stdlib import BoundLogger

app = ace.AceService(
    name=SERVICE_NAME,
    runners=[
        personalized_ranking.get_runner(),  # Price Parity uses the same model
        # Two Towers uses a custom model...
    ],
)

LOG: "BoundLogger" = structlog.get_logger()
BACKGROUND_TASK_RANDOM_WAIT_SEC: int = 30


async def check_all_execution_requirements(msg: str):
    # TODO: implement `check_execution_requirements()` for all other experiments
    await TWO_TOWERS_V2_EXPERIMENT.logic.Variation1.check_execution_requirements(msg=msg)  # V22
    await LogicTwoTowersV23.check_execution_requirements(msg=msg)  # V23
    await LogicTwoTowersV3.check_execution_requirements(msg=msg)  # V3
    await PRICE_PARITY_V2_EXPERIMENT.logic.Variation1.check_execution_requirements(msg=msg)


@app.on_background_startup()
async def setup_context():  # TODO Rename or split
    context = await Context.instance()
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
    await asyncio.gather(context.open(), data.refresh_kpp_scoring())
    set_context(context)
    LOG.info("Background tasks preparation DONE", debug=ace.DEBUG, stage=ace.STAGE)


@app.on_api_startup()
async def setup_db():  # DB connection pool (API thread)
    config = get_vendor_ranking_config().storage.postgres
    await db.init_connection_pool(service_name=SERVICE_NAME, config=config, query_timeout=config.main_query_timeout)
    await check_all_execution_requirements(msg="main")


@app.on_api_shutdown()
async def stop():
    if context := ctx():
        await context.close()
    await db.clear_connection_pool()


@app.background.task("every 55 minutes")
@newrelic.agent.background_task(name=f"{SERVICE_NAME}:refresh_vendors")
async def refresh_vendors() -> None:
    await random_wait(max_wait_sec=BACKGROUND_TASK_RANDOM_WAIT_SEC)
    await data.refresh_vendors()


@app.background.task("every 59 minutes")
@newrelic.agent.background_task(name=f"{SERVICE_NAME}:refresh_ranking_penalties")
async def refresh_ranking_penalties() -> None:
    await random_wait(max_wait_sec=BACKGROUND_TASK_RANDOM_WAIT_SEC)
    await data.refresh_ranking_penalties()


@app.background.task("every 59 minutes")
@newrelic.agent.background_task(name=f"{SERVICE_NAME}:refresh_ranking_penalties_v2")
async def refresh_ranking_penalties_v2() -> None:
    await random_wait(max_wait_sec=BACKGROUND_TASK_RANDOM_WAIT_SEC)
    await data.refresh_ranking_penalties_v2()


@app.background.task("every 120 minutes")
@newrelic.agent.background_task(name=f"{SERVICE_NAME}:refresh_chain_embeddings_two_towers_v2")
async def refresh_chain_embeddings() -> None:
    if (context := ctx()) and context.opened:
        await random_wait(max_wait_sec=BACKGROUND_TASK_RANDOM_WAIT_SEC)
        await context.reload_tt_models()
        LOG.info("TwoTowersV2: Chain embeddings refreshed successfully")
    else:
        LOG.warning("TwoTowersV2 Skip that embeddings refreshing, as context is not ready. Try next time.")


@app.ready.check()
def is_ready(_) -> bool:
    return data.is_loaded() and ctx() is not None


def _get_logic_for_experiment(fwf: FunWithFlags, experiment: ExperimentSettings) -> Optional["VendorLogicType"]:
    """Check if there is a logic for experiment specified in FWF."""
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


def _select_logic_for_country(country_code: str, config: LogicsConfig, logics_registry: dict[str, "VendorLogicType"]):
    default_logic = logics_registry[config.default]
    for logic_nickname, countries in config.countries_logic.items():
        if country_code in countries:
            if logic := logics_registry.get(logic_nickname):
                return logic
            # We can just count logs here (instead of writing custom metric), as this is not something "usual"
            LOG.warning(f"No logic found for {logic_nickname} in {country_code}. Use default.")
            return default_logic

    return default_logic


def select_default_logic(country_code: str, config: RankingLogicConfig, logics_registry: dict[str, "VendorLogicType"]):
    return _select_logic_for_country(country_code, config=config.logic.default, logics_registry=logics_registry)


def select_control_logic(country_code: str, config: RankingLogicConfig, logics_registry: dict[str, "VendorLogicType"]):
    return _select_logic_for_country(country_code, config=config.logic.control, logics_registry=logics_registry)


def select_holdout_logic(
    fwf: FunWithFlags,
    country_code: str,
    config: RankingLogicConfig,
    logics_registry: dict[str, Type["VendorLogicType"]]
):
    for config in config.logic.holdout:
        exp_name: str = config.experiment_name
        if active_experiment := fwf.active_experiments.get(exp_name):
            variation: str = active_experiment.variation
            if VARIATION1 == variation:  # for holdout, we check only variation 1, but not 2
                add_transaction_attr("fwf.experiment", exp_name, SERVICE_NAME)
                if logic := _select_logic_for_country(country_code, config=config, logics_registry=logics_registry):
                    LOG.debug(
                        f"Selected logic name={logic.NAME}, "
                        f"nickname={logic.NICKNAME or ''}, "
                        f"experiment={exp_name}, "
                        f"variation={variation}"
                    )
                    return logic
    return None


def validate_fwf(fwf: FunWithFlags, country_code: str, customer_id: int = 0):
    user_experiments = [
        (experiment_name, experiment)
        for experiment_name, experiment in fwf.active_experiments.items()
        if experiment.variation == VARIATION1
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
    config: RankingLogicConfig,
    logics_registry: dict[str, "VendorLogicType"],
    customer_id: Optional[int] = 0,
) -> "VendorLogicType":
    """
    Steps to choose logic according to FWF settings.
    1) If no FWF setting is provided, then execute default for a specific country.
    2) Check if there are any active experiments in FWF.
    3) Check if there are holdout experiments in FWF.
    4) Select Control logic.
    """
    if fwf is None or not fwf.active_experiments:
        return select_default_logic(country_code, config, logics_registry=logics_registry)

    validate_fwf(fwf=fwf, country_code=country_code, customer_id=customer_id)

    bahrain_and_oman = {"bh", "BH", "om", "OM"}
    for experiment, countries in (
        (CATBOOST_NO_FAST_SORT_EXPERIMENT, bahrain_and_oman),
        (PRICE_PARITY_EXPERIMENT, bahrain_and_oman),
        (PERSONALIZED_RANKING_TT_V3, None),
        (TWO_TOWERS_V2_EXPERIMENT, None),
        (KPP_TWO_TOWERS_V2_EXPERIMENT, None),
        (PRICE_PARITY_V2_EXPERIMENT, None),
    ):
        if countries and country_code not in countries:
            continue
        if logic := _get_logic_for_experiment(fwf, experiment):
            return logic

    if logic := select_holdout_logic(
        fwf=fwf, country_code=country_code, config=config, logics_registry=logics_registry
    ):
        return logic

    return select_control_logic(country_code, config, logics_registry=logics_registry)


@app.api.post("/v1/sort")
@app.api.post("/sort")
async def handler(
    request: VendorList,
    http_request: Request,
    background_tasks: fastapi.BackgroundTasks,
    context: Context = Depends(ctx),
    config: VendorRankingConfig = Depends(get_vendor_ranking_config),
    logics_registry: dict[str, "VendorLogicType"] = Depends(get_logics_registry),
) -> list[int]:
    logic: "VendorLogicType" = select_logic(
        fwf=request.fwf,
        country_code=request.location.country_code,
        config=config.ranking,
        logics_registry=logics_registry,
        customer_id=request.customer_id,
    )
    execution_log = LogEntry(SERVICE_NAME, await http_request.json())
    execution = logic(
        request=request,
        exec_log=execution_log,
        artifacts_service_registry=context.artifacts_service_registry,
    )
    newrelic.agent.set_transaction_name(execution.NAME)
    add_transaction_attrs(
        (
            # Can be extracted from the transaction name, but better to have it as a separate attribute also
            ("logic", logic.NAME),
            ("country_code", request.location.country_code.upper()),
            ("country", request.location.country_code.upper()),  # Backward compatibility with Two Towers
            ("customer_id", request.customer_id),
        ),
        service_name=SERVICE_NAME,
    )

    await execution.prepare_features()
    execution_log.response = final_sorting = await execution.sort()

    execution.push_transaction_stats()
    background_tasks.add_task(log_execution, execution_log)

    return final_sorting


if __name__ == "__main__":
    app.run()
