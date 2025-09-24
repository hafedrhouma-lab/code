#!/usr/bin/env python3

import fastapi
import newrelic.agent
from pydantic import conint

import ace
from ace.enums import CountryShortName
from ace.model_log import LogEntry, log_execution
from ace.storage import db
from item_lifecycle import SERVICE_NAME
from item_lifecycle.config.config import get_item_lifecycle_serving_config
from item_lifecycle.input import ItemReplenishmentResponse, ItemReplenishmentRequest
from item_lifecycle.item_replenishment import Logic

app = ace.AceService(SERVICE_NAME, runners=[])


@app.on_api_startup()
async def setup_db():  # DB connection pool (API thread)
    config = get_item_lifecycle_serving_config().storage.postgres
    await db.init_connection_pool(service_name=SERVICE_NAME, query_timeout=config.main_query_timeout, config=config)


@app.api.get(
    "/home/v1/{country_code}/customer/{customer_id}/item_replenishment", response_model=ItemReplenishmentResponse
)
async def handler(
    customer_id: conint(ge=0),
    country_code: CountryShortName,
    background_tasks: fastapi.BackgroundTasks,
) -> ItemReplenishmentResponse:
    # Do it automatically somehow?..
    newrelic.agent.set_transaction_name(f"{SERVICE_NAME}:item_replenishment")

    exec_log = LogEntry(
        SERVICE_NAME,
        {"customer_id": customer_id, "country_code": country_code},
    )

    request = ItemReplenishmentRequest(customer_id=customer_id, country_code=country_code)

    logic = Logic(request.customer_id, request.country_code, exec_log)
    response = await logic.fetch_categories()
    exec_log.response = response.dict()
    background_tasks.add_task(log_execution, exec_log)

    return response


if __name__ == "__main__":
    app.run()
