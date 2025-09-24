from typing import ClassVar, TYPE_CHECKING, Type

import polars as pl
import structlog
from pydantic import BaseModel

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

LOG: "BoundLogger" = structlog.get_logger()


# TODO provide actual implementation and move it to the right module
class MenuItemFeatures(BaseModel):
    items_id: int

    df_schema: ClassVar[dict[str, Type]] = {
        "items_id": int,
    }
    df_empty: ClassVar[pl.DataFrame] = pl.DataFrame(schema=df_schema)
