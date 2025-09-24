from typing import TYPE_CHECKING

import newrelic.agent
import structlog
from fastapi import Query, APIRouter, Depends
from langchain.schema import HumanMessage, AIMessage
from pydantic import conint
from sse_starlette.sse import ServerSentEvent

from ace.api.routes.route import get_empty_router
from ultron import SERVICE_NAME
from ultron import input
from ultron import logic
from ultron.api.v1.semantic_search.dependencies import get_semantic_cache_worker
from ultron.api.v1.semantic_search.models import (
    ItemsSemanticSearchRequest,
    SemanticSearchResponseDTO,
    SmartSemanticSearch,
    ConversationalSearch,
    SearchRecommendations,
    SmartSemanticSearchHome,
    QueryClassification,
)
from ultron.config.config import UltronServingConfig, get_ultron_serving_config
from ultron.runners.text_embeddings import TextEmbeddingsModelName
from ultron.vector_store.data import ShoppingItemData
from ultron.vector_store.store import VectorStore

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger
    from ultron.logic import Conversation

PING_EVENT = ServerSentEvent(data=input.PING_MESSAGE_EVENT.json())

LOG: "BoundLogger" = structlog.get_logger()


def build_router():
    """
    Returns: Router with endpoints for `semantic search` functionality.
    """
    router: APIRouter = get_empty_router()

    @router.post("/items-semantic-search", response_model=list[SemanticSearchResponseDTO])
    async def items_semantic_search(request: ItemsSemanticSearchRequest, page: conint(ge=1) = Query(default=1)):
        """
        Endpoint that takes items to search in and returns similar items
        """

        # Do it automatically somehow?..
        newrelic.agent.set_transaction_name(f"{SERVICE_NAME}:items-semantic-search")

        vector_store = VectorStore()
        similar_items: list[list[ShoppingItemData]] = await vector_store.get_similar_semantic_search_items(
            request=request, page=page, model_name=TextEmbeddingsModelName.ALL_MINILM_L6_V2
        )

        # pack the response
        original_search_terms = [item.search_term for item in request.queries]
        similar_items_list = [
            SemanticSearchResponseDTO(original_search=original_search_term, similar_items=similar)
            for original_search_term, similar in zip(original_search_terms, similar_items)
        ]

        return similar_items_list

    @router.post("/smart-query-understanding")
    async def smart_query_understanding(
        request: SmartSemanticSearch,
        config: UltronServingConfig = Depends(get_ultron_serving_config),
    ):
        # Do it automatically somehow?..
        newrelic.agent.set_transaction_name(f"{SERVICE_NAME}:smart-query-understanding")

        conv: "Conversation" = logic.for_smart_semantic_search(
            request.version, openai_api_key=config.external_apis.openai_api_key
        )

        dialog = [HumanMessage(content=request.query), AIMessage(content="```json\n[")]

        response = input.from_langchain(await conv.generate(dialog))

        response.content = "```json\n[" + response.content

        LOG.debug(f"Response content: {response.content}")

        return {'message': response.content}
    

    @router.post("/smart-semantic-search")
    async def smart_semantic_search(request: SmartSemanticSearch):
        # Do it automatically somehow?..
        newrelic.agent.set_transaction_name(f"{SERVICE_NAME}:smart-semantic-search")

        json_content = await smart_query_understanding(request)

        combined_json_content = ItemsSemanticSearchRequest(
            queries=json_content,
            chains=request.chains,
            page_size=request.k,
            customer_id=request.customer_id,
            country_code=request.country_code,
        )

        items = await items_semantic_search(combined_json_content)

        return items

    @router.post("/conversational-query-understanding")
    async def conversational_query_understanding(request: ConversationalSearch):
        # Do it automatically somehow?..
        newrelic.agent.set_transaction_name(f"{SERVICE_NAME}:conversational-query-understanding")

        response = await logic.for_conversational_semantic_search(request.version, request.dialog)

        if response["content"] is not None:
            return response
        else:
            function_call = response["function_call"]
            function_args = function_call["arguments"]

            json_content = logic.extract_and_parse_json(function_args)

            return json_content

    @router.post("/conversational-search")
    async def conversational_search(request: ConversationalSearch):
        # Do it automatically somehow?..
        newrelic.agent.set_transaction_name(f"{SERVICE_NAME}:conversational-search")

        response: dict = await conversational_query_understanding(request)

        if "content" in response:
            return response
        else:
            json_content = response["items_to_search"]

            combined_json_content = ItemsSemanticSearchRequest(
                queries=json_content,
                chains=request.chains,
                page_size=request.k,
                customer_id=request.customer_id,
                country_code=request.country_code,
            )

            items = await items_semantic_search(combined_json_content)

            return items

    @router.post("/search-recommendations")
    async def search_recommendations(
        request: SearchRecommendations,
        config: UltronServingConfig = Depends(get_ultron_serving_config),
    ):
        # Do it automatically somehow?..
        newrelic.agent.set_transaction_name(f"{SERVICE_NAME}:search-recommendations")

        conv: "Conversation" = logic.for_search_recommendations(
            request.version, openai_api_key=config.external_apis.openai_api_key
        )
        response = await conv.generate(messages=[HumanMessage(content=request.query)])
        recommendations = logic.extract_and_parse_json(response.content)
        return recommendations

    @router.post("/smart-query-understanding-home", response_model=QueryClassification)
    async def smart_query_understanding_home(
        request: SmartSemanticSearchHome,
        config: UltronServingConfig = Depends(get_ultron_serving_config),
        semantic_cache_worker: logic.SemanticCacheWorker = Depends(get_semantic_cache_worker),
    ):
        # Do it automatically somehow?..
        newrelic.agent.set_transaction_name(f"{SERVICE_NAME}:smart-query-understanding-home")

        content = logic.get_value_from_json_squ(request.query)

        LOG.debug(f"Response content: {content}")
        return logic.convert_to_json(request.query, content)

    return router
