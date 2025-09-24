import datetime as dt
from typing import Any, Optional, TYPE_CHECKING, ClassVar

import structlog
from pydantic import BaseModel, Field
from starlette_context import _request_scope_context_storage
from starlette_context import context as current_request_context
from starlette_context.errors import ContextDoesNotExistError

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

LOG: "BoundLogger" = structlog.get_logger()


class APIStatName:
    DT_START = "dt_start"
    RAW_BODY = "raw_body"


class APICallStats(BaseModel):
    dt_start: Optional[dt.datetime] = Field(alias=APIStatName.DT_START)
    raw_body: Optional[bytes] = Field(alias=APIStatName.RAW_BODY)

    STATS_KEY: ClassVar[str] = "stats"

    @classmethod
    def get_request_current_duration_seconds(cls) -> Optional[float]:
        if dt_start := cls.get_value(APIStatName.DT_START):
            return (cls.utc_now() - dt_start).total_seconds()
        return None

    @classmethod
    def utc_now(cls) -> dt.datetime:
        return dt.datetime.utcnow()

    @classmethod
    def get_from_current_request_context(cls) -> "APICallStats":
        return current_request_context.get(cls.STATS_KEY)

    @classmethod
    def get_value(cls, name: str):
        try:
            data = cls.get_from_current_request_context()
            return getattr(data, name)
        except ContextDoesNotExistError:
            pass

    @classmethod
    def set_raw_request_body(cls, body: bytes):
        cls.set_value(APIStatName.RAW_BODY, body)

    @classmethod
    def set_value(cls, name: str, value: Any):
        try:
            data = current_request_context.data
            setattr(data[cls.STATS_KEY], name, value)
            _request_scope_context_storage.set(data)
        except ContextDoesNotExistError:
            pass

    @classmethod
    def init_current_request_context(cls):
        try:
            data = current_request_context.data
            kwargs = {APIStatName.DT_START: cls.utc_now()}
            data[cls.STATS_KEY] = cls(**kwargs)
            _request_scope_context_storage.set(data)
        except ContextDoesNotExistError:
            pass

    class Config:
        allow_population_by_field_name = True
