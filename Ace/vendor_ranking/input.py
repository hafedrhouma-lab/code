from typing import Optional, ClassVar, Type

import newrelic.agent
import polars as pl
from pydantic import BaseModel

from abstract_ranking.input import AbstractRankingRequest


@newrelic.agent.function_trace()
def split_by_vendor_availability(features: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Splits features dataframe by vendor availability, returns a tuple (available, busy).
    """
    features = features.with_columns(pl.when(pl.col("status") == "0").then(True).otherwise(False).alias("is_available"))
    dfs_by_availability = features.partition_by(by="is_available", as_dict=True, maintain_order=True)

    return (
        dfs_by_availability.get(True, VendorFeatures.df_empty),
        dfs_by_availability.get(False, VendorFeatures.df_empty),
    )


class VendorFeatures(BaseModel):
    vendor_id: int
    chain_id: Optional[int] = None
    delivery_fee: Optional[float] = None
    delivery_time: Optional[float] = None
    vendor_rating: Optional[float] = None
    status: Optional[str] = "2"  # If missing, assume busy
    min_order_amount: Optional[float] = 0.0
    has_promotion: Optional[bool] = False

    # TODO Get types (and names?) from above...
    df_schema: ClassVar[dict[str, Type]] = {
        "chain_id": int,
        "vendor_id": int,
        "delivery_fee": float,
        "delivery_time": float,
        "vendor_rating": float,
        "status": str,
        "min_order_amount": float,
        "has_promotion": bool,
    }
    df_empty: ClassVar[pl.DataFrame] = pl.DataFrame(schema=df_schema)


class VendorList(AbstractRankingRequest):
    # vendors: list[VendorFeatures] = []
    vendors: list[dict] = []
    vendors_df: dict[str, list] = {}  # TODO Migrate to this new input format (columnar)
