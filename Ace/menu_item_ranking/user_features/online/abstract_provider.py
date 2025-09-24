from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, TYPE_CHECKING

import pydantic as pd

from abstract_ranking.two_tower import TTVersion

if TYPE_CHECKING:
    from menu_item_ranking.request.input import MenuItemRequest


@dataclass
class AbstractUserOnlineFeaturesProvider(ABC):
    version: TTVersion

    @abstractmethod
    async def get_features(self, request: "MenuItemRequest") -> pd.BaseModel:
        pass


BaseUserOnlineFeaturesProvider = TypeVar("BaseUserOnlineFeaturesProvider", bound=AbstractUserOnlineFeaturesProvider)
