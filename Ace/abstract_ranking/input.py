from datetime import datetime
from typing import Optional

from pydantic import BaseModel, constr, Field


class Location(BaseModel):
    country_id: int
    country_code: constr(to_upper=True)
    city_id: Optional[int]
    area_id: int
    latitude: float
    longitude: float


class Experiment(BaseModel):
    variation: str
    abTest: Optional[bool] = None


class FunWithFlags(BaseModel):
    active_experiments: dict[str, Experiment] = Field(default_factory=dict)

    def get(self, name: str) -> Experiment | None:
        return None  # TODO An empty one by default


_EMPTY_FWF = FunWithFlags()


class AbstractRankingRequest(BaseModel):
    fwf: Optional[FunWithFlags] = Field(default_factory=lambda: _EMPTY_FWF)
    timestamp: datetime
    perseusSessionId: Optional[str]
    customer_id: Optional[int] = Field(alias="account_id")
    device_source: int
    app_version: str
    locale: str
    location: Optional[Location]

    class Config:
        allow_population_by_field_name = True
