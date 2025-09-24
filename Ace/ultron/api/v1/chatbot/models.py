from enum import Enum

from pydantic import conlist, BaseModel


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"

    # A quick fix for the wrong input data
    SERVER = "server"


class Message(BaseModel):
    role: Role = Role.USER
    content: str


class RecipeChat(BaseModel):
    dialog: conlist(Message, min_items=1)
    version: str = "v025"
    intent_check: bool = True
    stream: bool = False


class SupportChat(BaseModel):
    dialog: conlist(Message, min_items=1)
    version: str = "v02"
