from typing import TYPE_CHECKING, Union, Optional

import newrelic.agent
import structlog
from pydantic import confloat, constr
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from ace.api.middlewares.stats.model import APICallStats
from ace.api.routes.match import RouterPathMatcher
from ace.newrelic import CustomRequestAttrs

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response, StreamingResponse
    from structlog.stdlib import BoundLogger
    from starlette.types import ASGIApp
    from starlette.middleware.base import DispatchFunction

LOG: "BoundLogger" = structlog.get_logger()


class ReportLongRequestsMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        service_name: str,
        request_time_limit: float,
        app: "ASGIApp",
        dispatch: Optional["DispatchFunction"] = None,
    ):
        super().__init__(app=app, dispatch=dispatch)
        self.request_time_limit: float = confloat(gt=0)(request_time_limit)
        self.service_name: str = constr(min_length=1)(service_name)

    async def dispatch(self, request: "Request", call_next: "RequestResponseEndpoint") -> "Response":
        APICallStats.init_current_request_context()
        response: "Response" = await call_next(request)
        self.send_request_stats(request=request, response=response)
        return response

    @newrelic.agent.function_trace()
    def send_request_stats(self, request: "Request", response: Union["Response", "StreamingResponse"]):
        stats: APICallStats = APICallStats.get_from_current_request_context()
        if (duration := stats.get_request_current_duration_seconds()) <= self.request_time_limit:
            return

        params = {
            CustomRequestAttrs.REQUEST: stats.raw_body,
            CustomRequestAttrs.METHOD: request.method,
            CustomRequestAttrs.STATUS: response.status_code,
            CustomRequestAttrs.PATH: RouterPathMatcher.get_router_path_from_request(request),
            CustomRequestAttrs.SERVICE: self.service_name,
            CustomRequestAttrs.DURATION: duration,
        }
        newrelic.agent.record_custom_event("Ace/LongRequest", params)
