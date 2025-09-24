from dataclasses import dataclass
from typing import TYPE_CHECKING, Type, ClassVar

import structlog

from abstract_ranking.context import ArtefactsServiceRegistry, TTConfig
from abstract_ranking.two_tower import TTVersion
from abstract_ranking.two_tower.artefacts_service import ArtefactsService, prepare_configs
from menu_item_ranking.artefacts_service import MenuItemArtefactsService

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

LOG: "BoundLogger" = structlog.get_logger()


@dataclass
class MenuArtefactsServiceRegistry(ArtefactsServiceRegistry):
    configs: ClassVar[dict[TTVersion, TTConfig]] = {
        TTVersion.MENUITEM_V1: TTConfig(
            {"EG", "OM", "QA"},
            prepare_configs(
                version=TTVersion.MENUITEM_V1,
                configs=[
                    ("EG", 3905),
                    ("OM", 4441),
                    ("QA", 3871),
                ]
            )
        )
    }

    @classmethod
    def get_artefacts_service_type(cls, version: TTVersion) -> Type[ArtefactsService]:
        match version:
            case TTVersion.MENUITEM_V1:
                return MenuItemArtefactsService
            case TTVersion.MENUITEM_V2_BIG:
                return MenuItemArtefactsService
            case _:
                raise ValueError(f"Model version {version} is not supported for menu items ranking")
