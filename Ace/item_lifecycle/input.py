from typing import Optional

from pydantic import BaseModel, conint

from ace.enums import CountryShortName


class ItemReplenishmentResponse(BaseModel):
    customer_id: int
    country_code: str
    item_replenishment_categories: list[Optional[str]]


class ItemReplenishmentRequest(BaseModel):
    customer_id: conint(ge=0)
    country_code: CountryShortName
