from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, TYPE_CHECKING

import pandas as pd
from typing_extensions import Self

from abstract_ranking.two_tower import TTVersion
from ace.enums import CountryShortNameUpper
from menu_item_ranking.artefacts_service_registry import MenuArtefactsServiceRegistry

if TYPE_CHECKING:
    from menu_item_ranking.request.input import MenuItemRequest


@dataclass
class AbstractUserOfflineFeaturesProvider(ABC):

    @classmethod
    @abstractmethod
    async def instance(
        cls,
        artefacts_service_registry: MenuArtefactsServiceRegistry,
        version: TTVersion,
        country: CountryShortNameUpper,
        features_names: list[str],
        *args,
        **kwargs
    ) -> Self:
        pass

    @abstractmethod
    async def get_features(self, request: "MenuItemRequest") -> pd.DataFrame:
        pass


UserOfflineFeaturesProvider = TypeVar("UserOfflineFeaturesProvider", bound=AbstractUserOfflineFeaturesProvider)
