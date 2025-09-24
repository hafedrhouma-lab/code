from dataclasses import dataclass
from typing import TYPE_CHECKING

import pydantic as pd

from abstract_ranking.two_tower import TTVersion
from menu_item_ranking.user_features.online.abstract_provider import (
    AbstractUserOnlineFeaturesProvider
)
from menu_item_ranking.user_features.online.features import UserOnlineFeatures

if TYPE_CHECKING:
    from menu_item_ranking.request.input import MenuItemRequest


@dataclass
class UserOnlineFeaturesProvider(AbstractUserOnlineFeaturesProvider):
    version: TTVersion = TTVersion.MENUITEM_V1

    async def get_features(self, request: "MenuItemRequest") -> pd.BaseModel:
        return UserOnlineFeatures.from_request(request=request)
