from fastapi import APIRouter

from ultron.api.v1.chatbot.router import build_router as build_chatbot_router
from ultron.api.v1.items_to_purchase.router import build_router as build_items_to_purchase_router
from ultron.api.v1.semantic_search.router import build_router as build_semantic_search_router


def build_router_v1() -> APIRouter:
    router = APIRouter()
    router.include_router(build_items_to_purchase_router(), tags=["Items to purchase"])
    router.include_router(build_semantic_search_router(), tags=["Semantic search"])
    router.include_router(build_chatbot_router(), tags=["Chatbot"])
    return router
