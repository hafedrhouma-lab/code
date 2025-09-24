from fastapi import APIRouter

from ultron.api.v1 import build_router_v1


def build_main_router() -> APIRouter:
    router = APIRouter()
    router.include_router(build_router_v1(), prefix="/v1")
    return router
