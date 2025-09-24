from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from abstract_ranking.two_tower import TTVersion
from menu_item_ranking.user_features.offline.abstract_provider import AbstractUserOfflineFeaturesProvider

if TYPE_CHECKING:
    from menu_item_ranking.request.input import MenuItemRequest


@dataclass
class FeatureStoreUserOfflineFeaturesProvider(AbstractUserOfflineFeaturesProvider):
    version: TTVersion

    async def get_features(self, request: "MenuItemRequest") -> pd.DataFrame:
        pass
