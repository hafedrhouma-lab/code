from datetime import timezone, datetime
from typing import TYPE_CHECKING

import pydantic as pd
from pygeohash_fast import encode as encode_geohash
from typing_extensions import Self

if TYPE_CHECKING:
    from abstract_ranking.input import Location
    from vendor_ranking.input import VendorList


class UserStaticFeaturesV2(pd.BaseModel):
    """ All static user features must be not nulls. """
    account_id: int
    country_iso: str = pd.Field(alias="country_code")
    most_recent_10_clicks_wo_orders: str = pd.Field(alias="prev_clicks")
    most_recent_10_orders: str = pd.Field(alias="user_prev_chains")
    frequent_clicks: str = pd.Field(alias="freq_clicks")
    frequent_chains: str = pd.Field(alias="freq_chains")
    most_recent_15_search_keywords: str = pd.Field(alias="prev_searches")

    class Config:
        allow_population_by_field_name = True


class UserStaticFeaturesV3(UserStaticFeaturesV2):
    """ All static user features must be not nulls. """
    account_order_source: str
    account_log_order_cnt: float
    account_log_avg_gmv_eur: float
    account_incentives_pct: float
    account_is_tpro: float
    account_discovery_pct: float

    class Config:
        allow_population_by_field_name = True


class UserDynamicFeatures(pd.BaseModel):
    order_hour: int = pd.Field(ge=0, le=23, description="hour (0-23)")
    order_weekday: int = pd.Field(ge=0, le=6, description="day of the week, where Monday == 0 ... Sunday == 6")
    geohash6: str
    delivery_area_id: int

    @classmethod
    def from_request(cls, request: "VendorList") -> Self:
        utc_ts = request.timestamp.astimezone(timezone.utc)
        return cls(
            order_hour=utc_ts.hour,
            order_weekday=utc_ts.weekday(),
            geohash6=encode_geohash(lng=float(request.location.longitude), lat=float(request.location.latitude), len=6),
            delivery_area_id=request.location.area_id,
        )

    @classmethod
    def from_location(cls, location: "Location") -> Self:
        utc_ts_now = datetime.utcnow().astimezone(timezone.utc)
        return cls(
            order_hour=utc_ts_now.hour,
            order_weekday=utc_ts_now.weekday(),
            geohash6=encode_geohash(lng=float(location.longitude), lat=float(location.latitude), len=6),
            delivery_area_id=location.area_id,
        )


class UserDynamicFeaturesV3(UserDynamicFeatures):
    delivery_area_id: str
