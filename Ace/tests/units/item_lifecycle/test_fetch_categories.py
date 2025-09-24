import pytest

from item_lifecycle.item_replenishment import Logic
from item_lifecycle.input import ItemReplenishmentResponse
from ace.model_log import LogEntry
from item_lifecycle import SERVICE_NAME


@pytest.mark.asyncio
@pytest.mark.parametrize("customer_id, country_code", [(123, "ae")])
async def test_fetch_categories(setup_db, customer_id, country_code):
    exec_log = LogEntry(
        SERVICE_NAME,
        {"customer_id": customer_id, "country_code": country_code},
    )
    logic = Logic(customer_id, country_code, exec_log)
    response = await logic.fetch_categories()

    assert isinstance(response, ItemReplenishmentResponse)
    assert response.customer_id == customer_id
    assert response.country_code == country_code
