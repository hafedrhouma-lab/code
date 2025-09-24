from typing import Type, Optional

from pydantic import BaseModel, parse_obj_as, Field

from menu_item_ranking.logic.base_logic import MenuItemBaseLogic

LogicType = Type[MenuItemBaseLogic]


class ExperimentLogic(BaseModel):
    Control: Optional[LogicType] = Field(default=None, const=True)
    Variation1: LogicType
    Variation2: Optional[LogicType]
    Variation3: Optional[LogicType]

    class Config:
        arbitrary_types_allowed = True


class ExperimentSettings(BaseModel):
    name: str
    logic: ExperimentLogic


def as_experiment_settings(settings: dict) -> ExperimentSettings:
    return parse_obj_as(ExperimentSettings, settings)


# Mapping from logic's nickname to logic's class
LOGICS_REGISTRY: dict[str, Type[MenuItemBaseLogic]] = {
    "default": MenuItemBaseLogic
}


def get_logics_registry():
    global LOGICS_REGISTRY
    return LOGICS_REGISTRY
