from json import JSONDecodeError

import orjson
import structlog
from fastapi.routing import APIRoute, APIRouter
from fastapi import HTTPException
from typing import Callable, Any, Coroutine, TYPE_CHECKING, Union

from ace.api.middlewares.stats.model import APICallStats
from ace.newrelic import CustomRequestAttrs, add_transaction_attr

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger
    from fastapi import Request, Response

LOG: "BoundLogger" = structlog.get_logger()


def parse_body(raw_body: bytes) -> Union[dict, bytes]:
    """
    Extract request body. Return pure bytes if body is not a json.
    """
    try:
        return orjson.loads(raw_body)
    except JSONDecodeError:
        return raw_body


class ErrorHandlingRoute(APIRoute):
    """Route which logs request's payload on any exception."""

    def get_route_handler(self) -> Callable:
        handler: Callable[["Request"], Coroutine[Any, Any, "Response"]] = super().get_route_handler()

        async def custom_route_handler(request: "Request") -> "Response":
            APICallStats.set_raw_request_body(raw_body := await request.body())
            try:
                return await handler(request)
            except Exception as ex:
                request_body = parse_body(raw_body)
                if isinstance(ex, HTTPException):
                    ex_msg = f"code={ex.status_code}, detail={ex.detail}"
                else:
                    ex_msg = str(ex)
                LOG.error(f"API request failed. Error: {ex_msg}. RequestBody: {request_body}")
                add_transaction_attr(CustomRequestAttrs.REQUEST, request_body)
                raise

        return custom_route_handler


def get_empty_router() -> APIRouter:
    return APIRouter(route_class=ErrorHandlingRoute)
