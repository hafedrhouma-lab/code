from enum import Enum
from typing import Iterable, Optional, AsyncIterable, List

import newrelic.agent
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, conlist, validator

from ultron.api.v1.chatbot.models import Role, Message


class UserIntents(BaseModel):
    user_intents: List[str]


class SortBy(BaseModel):
    number_of_orders: Optional[str]
    price: Optional[str]

    @validator("number_of_orders", "price")
    def validate_values(cls, v):
        assert v in ["asc", "desc"], 'Value must be either "asc" or "desc"'
        return v


class ComparisonSign(str, Enum):
    less_than = "<"
    greater_than = ">"
    equals = "="


class FilterBy(BaseModel):
    comparison_sign: ComparisonSign
    price: str


class RecipeMissingItems(BaseModel):
    basket_items: conlist(str, min_items=1)
    stream: bool = False


class MessageType(Enum):
    MESSAGE = "message"
    PING = "ping"


class MessageEvent(BaseModel):
    class Data(BaseModel):
        output_adf: str
        finish_reason: Optional[str] = None
        event_type: MessageType = MessageType.MESSAGE

    data: Data


PING_MESSAGE_EVENT = MessageEvent(data=MessageEvent.Data(output_adf="", event_type=MessageType.PING))


def to_langchain(dialog: Iterable[Message]) -> Iterable[BaseMessage]:
    for m in dialog:
        if m.role == Role.USER:
            yield HumanMessage(content=m.content)
        else:
            yield AIMessage(content=m.content)


def from_langchain(message: BaseMessage) -> Message:
    role = Role.USER if isinstance(message, HumanMessage) else Role.ASSISTANT

    return Message(role=role, content=message.content)


@newrelic.agent.function_trace()
async def wrap_langchain_stream(response_stream: AsyncIterable) -> AsyncIterable[str]:
    # TODO Handle openai.error:RateLimitError
    async for content, finish_reason in response_stream:
        chat_event = MessageEvent(data=MessageEvent.Data(finish_reason=finish_reason, output_adf=content))
        yield chat_event.json()
