import datetime as dt
from typing import Optional

import pydantic as pd
from pydantic import conint
from typing_extensions import Self

from abstract_ranking.input import Location
from menu_item_ranking.request.input import MenuItemRequest


class UserOnlineFeatures(pd.BaseModel):
    order_hour: conint(ge=0, le=23) = pd.Field(description="hour (0-23)")
    order_weekday: conint(ge=0, le=6) = pd.Field(description="day of the week, where Monday == 0 ... Sunday == 6")
    delivery_area_id: conint(ge=0)
    chain_id: Optional[str]

    @classmethod
    def from_request(cls, request: "MenuItemRequest") -> Self:
        return cls.from_location(request.location)

    @classmethod
    def from_location(cls, location: "Location") -> Self:
        utc_ts_now = dt.datetime.utcnow().astimezone(dt.timezone.utc)
        return cls(
            order_hour=utc_ts_now.hour,
            order_weekday=utc_ts_now.weekday(),
            delivery_area_id=location.area_id,
            chain_id=None
        )
