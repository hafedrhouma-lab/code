from enum import Enum
from typing import Optional

from pydantic import conlist, BaseModel, Field, validator

from ultron.api.v1.chatbot.models import Message
from ultron.vector_store.data import ShoppingItemData


class Verticals(str, Enum):
    FOOD = "food"
    GROCERY = "grocery"


class VerticalsSearch(str, Enum):
    FOOD = "food"
    GROCERY = "grocery"
    HEALTH_AND_BEAUTY = "health & beauty"
    FLOWERS = "flowers"
    MORE_SHOPS = "more shops"


class QueryClassification(BaseModel):
    query: str
    category: conlist(VerticalsSearch, min_items=1)


class SortOrder(str, Enum):
    asc = "asc"
    desc = "desc"


class SemanticSearchItem(BaseModel):
    search_term: str
    verticals: list[Verticals]
    sort_by: Optional[dict[str, SortOrder]]
    filter_by: Optional[dict[str, str]]
    past_orders_only: Optional[bool]

    @validator("verticals", always=True)
    def validator_verticals(cls, value):
        if isinstance(value, list):
            return value
        return [value]


class ItemsSemanticSearchRequest(BaseModel):
    queries: conlist(SemanticSearchItem, min_items=1) = Field(alias="json_content")
    chains: Optional[list[int]]
    customer_id: int
    page_size: int = Field(default=3, alias="k")
    country_code: str = "ae"

    class Config:
        allow_population_by_field_name = True


class SemanticSearchResponseDTO(BaseModel):
    original_search: str
    similar_items: list[ShoppingItemData]


class SmartSemanticSearch(BaseModel):
    query: str
    chains: Optional[list[int]]
    customer_id: int
    k: int
    version: str = "v04"
    country_code: Optional[str] = "ae"


class SmartSemanticSearchHome(BaseModel):
    query: str
    version: str = "v01"


class ConversationalSearch(BaseModel):
    dialog: conlist(Message, min_items=1)
    chains: Optional[list[int]]
    customer_id: int
    k: int
    version: str = "v01"
    country_code: Optional[str] = "ae"


class SearchRecommendations(BaseModel):
    query: str
    version: str = "v01"
