from typing import TYPE_CHECKING

from starlette_context.plugins import RequestIdPlugin

from ace.api.middlewares.stats.middleware import ReportLongRequestsMiddleware

if TYPE_CHECKING:
    from fastapi import FastAPI


def add_middlewares(api: "FastAPI", service_name: str):
    # 1) --- API Request stats ---
    api.add_middleware(ReportLongRequestsMiddleware, service_name=service_name, request_time_limit=5)

    # 0) --- ContextVar for request ---
    from starlette_context.middleware import ContextMiddleware

    api.add_middleware(ContextMiddleware, plugins=(RequestIdPlugin(),))
