from typing import Optional

from pydantic import Field

from abstract_ranking.input import AbstractRankingRequest


class MenuItemRequest(AbstractRankingRequest):
    vendor_id: int = Field(alias="branch_id")
    chain_id: Optional[int] = Field(alias="chain_id", default=None)

    class Config:
        allow_population_by_field_name = True
