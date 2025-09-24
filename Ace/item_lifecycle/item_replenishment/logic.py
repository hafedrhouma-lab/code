import newrelic.agent
import structlog

from item_lifecycle import SERVICE_NAME
from item_lifecycle.input import ItemReplenishmentResponse
from ace.storage import db
from ace.newrelic import add_transaction_attrs
from ace.model_log import LogEntry

logger = structlog.get_logger()


class Logic:
    """
    Logic class for handling item replenishment data.

    Attributes:
        ITEM_REPLENISHMENT_QUERY (str): SQL query for fetching item replenishment data.
        customer_id (int): The customer ID.
        country_code (str): The country code.
        exec_log (LogEntry): Log entry for execution details.
    """

    ITEM_REPLENISHMENT_QUERY = """SELECT * FROM item_replenishment WHERE country_code = $1 AND account_id = $2;"""

    def __init__(self, customer_id: int, country_code: str, exec_log: LogEntry) -> None:
        """
        Initializes the Logic instance.

        Args:
            customer_id (int): The customer ID.
            country_code (str): The country code.
            exec_log (LogEntry): Log entry for execution details.
        """
        super().__init__()
        self.customer_id = customer_id
        self.country_code = country_code

        exec_log.model_tag = "item_replenishment"
        self.exec_log = exec_log

    @newrelic.agent.function_trace()
    async def fetch_categories(self) -> ItemReplenishmentResponse:
        """
        Fetches item replenishment categories for a customer.

        Returns:
            ItemReplenishmentResponse: The response containing item replenishment data.

        Example:
        {
          "customer_id": 23735891,
          "country_code": "ae",
          "item_replenishment_categories": [
            "1035fc72-c657-4bf4-a2c5-968618ec8b89",
            "1c8fb3a0-a7dd-4d52-9299-0ad2995b90be",
            "429b86ed-9b1f-4ae0-a8a0-621d45052ac8",
            "7ab2fa01-af6d-4dce-9413-968c9ad5251e",
            "8ea5bf94-9b85-46a8-86f9-8e7603691638",
            "e3582e76-4300-4d75-b7eb-f2022bb7f663",
            "1e57b123-dd6c-4ef8-a553-3655ae8bf49e"
          ]
        }

        Note: These are the categories the customer is eligible for on Tmart at the time of querying the data.
        Categories id are the ones used by DH.
        """
        async with db.connections().acquire() as conn:
            item_replenishment_data = await db.fetch_row_as_dict(
                conn, Logic.ITEM_REPLENISHMENT_QUERY, self.country_code, self.customer_id
            )

            categories_list = item_replenishment_data.get("category_id_list", [])

            add_transaction_attrs((("country_code", self.country_code),), service_name=SERVICE_NAME)
            if not categories_list:
                add_transaction_attrs((("customer_id", self.customer_id),), service_name=SERVICE_NAME)

            response = ItemReplenishmentResponse(
                customer_id=self.customer_id,
                country_code=self.country_code,
                item_replenishment_categories=categories_list,
            )

            return response
