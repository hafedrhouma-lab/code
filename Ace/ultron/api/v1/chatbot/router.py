import newrelic.agent
from fastapi import HTTPException, APIRouter, Depends
from langchain.schema import HumanMessage, AIMessage
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from ace.api.routes.route import get_empty_router
from ultron import SERVICE_NAME
from ultron import input
from ultron import logic
from ultron.api.v1.chatbot.models import Message, RecipeChat, SupportChat
from ultron.config.config import UltronServingConfig, get_ultron_serving_config

PING_EVENT = ServerSentEvent(data=input.PING_MESSAGE_EVENT.json())


def build_router() -> APIRouter:
    router: APIRouter = get_empty_router()

    def ping_event():
        return PING_EVENT

    def sse_response(generator):
        return EventSourceResponse(generator, ping_message_factory=ping_event)

    @router.post("/recipe-chatbot", response_model=Message)
    async def recipe_chatbot(
        request: RecipeChat,
        config: UltronServingConfig = Depends(get_ultron_serving_config),
    ):
        # Do it automatically somehow?..
        newrelic.agent.set_transaction_name(f"{SERVICE_NAME}:recipe-chatbot")

        dialog = list(input.to_langchain(request.dialog))
        last_chat_message = dialog[-1]
        last_request = request.dialog[-1]
        openai_api_key = config.external_apis.openai_api_key
        logic.check_moderation([last_request], openai_api_key=openai_api_key)
        if (
            (
                request.intent_check
                and await logic.check_malicious_intent([last_chat_message], openai_api_key=openai_api_key)
            )
            or logic.check_for_keywords([last_chat_message])
            or logic.check_for_adversarial_attack([last_request])
        ):
            raise HTTPException(status_code=400, detail="Malicious intent detected")

        request.version = "v030"
        conv = logic.for_recipe(request.version, openai_api_key=openai_api_key)

        messages = []
        for m in dialog:
            m.content = m.content[:100]
            messages.append(m)
        messages = logic.get_unique_messages(messages)[-4:]
        messages.append(HumanMessage(content="and give shopping list if appropriate"))
        messages.append(HumanMessage(content="do not reply anything unrelated to food, groceries"))
        messages.append(
            HumanMessage(
                content="make sure to add $$$ items_to_purchase_for_recipe: "
                "... $$$ if it is relevant because user is asking for a recipe or ingredients"
            )
        )

        response = (
            sse_response(input.wrap_langchain_stream(conv.generate_stream(messages)))
            if request.stream
            else input.from_langchain(await conv.generate(messages))
        )

        return response

    @router.post("/support-chatbot")
    async def support_chatbot(
        request: SupportChat,
        config: UltronServingConfig = Depends(get_ultron_serving_config),
    ):
        # Do it automatically somehow?..
        newrelic.agent.set_transaction_name(f"{SERVICE_NAME}:support-chatbot")

        conv, support_output_parser = logic.for_support_chatbot(
            request.version, openai_api_key=config.external_apis.openai_api_key
        )

        dialog = list(input.to_langchain(request.dialog))

        dialog.append(AIMessage(content="```json\n{"))

        response = input.from_langchain(await conv.generate(dialog))

        response.content = "```json\n{" + response.content

        json_content = logic.extract_and_parse_for_support_chatbot(response.content, support_output_parser)

        return json_content

    return router
