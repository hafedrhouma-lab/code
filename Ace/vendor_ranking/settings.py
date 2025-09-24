from typing import Type, Optional

from pydantic import BaseModel, parse_obj_as, Field

from vendor_ranking import (
    personalized_ranking,
    price_parity,
    price_parity_v2,
    no_fast_sort_ranking,
)
from vendor_ranking.base_logic import VendorBaseLogic
from vendor_ranking.key_preferred_partner.logic import KeyPreferredPartner
from vendor_ranking.price_parity.logic import (
    TTv2PriceParity,
    TTv3PriceParity,
    TTv23PriceParity,
)
from vendor_ranking.two_tower.logic import (
    LogicTwoTowersV22,
    LogicTwoTowersV3,
    LogicTwoTowersV23,
)

VendorLogicType = Type[VendorBaseLogic]

VARIATION1 = "Variation1"
VARIATION2 = "Variation2"


class ExperimentLogic(BaseModel):
    Control: Optional[VendorLogicType] = Field(default=None, const=True)
    Variation1: VendorLogicType
    Variation2: Optional[VendorLogicType]
    Variation3: Optional[VendorLogicType]

    class Config:
        arbitrary_types_allowed = True


class ExperimentSettings(BaseModel):
    name: str
    logic: ExperimentLogic


def as_experiment_settings(settings: dict) -> ExperimentSettings:
    return parse_obj_as(ExperimentSettings, settings)


PRICE_PARITY_EXPERIMENT: ExperimentSettings = as_experiment_settings(
    {
        "name": "exp_price_parity_ranking",
        "logic": {
            VARIATION1: price_parity.Logic,  # catboost + fast sort + penalization
        },
    }
)

PERSONALIZED_RANKING_TT_V3 = as_experiment_settings(
    {
        "name": "exp_personalized_ranking_tt_v3",
        "logic": {VARIATION1: LogicTwoTowersV3, VARIATION2: TTv3PriceParity},
    }
)

PRICE_PARITY_V2_EXPERIMENT: ExperimentSettings = as_experiment_settings(
    {
        "name": "exp_price_parity_ranking_v2",
        "logic": {
            VARIATION1: price_parity_v2.Logic,  # tt v22 + penalization
        },
    }
)

TWO_TOWERS_V2_EXPERIMENT = as_experiment_settings(
    {
        "name": "exp_personalized_ranking_v4",
        "logic": {
            VARIATION1: LogicTwoTowersV22,
        },
    }
)

KPP_TWO_TOWERS_V2_EXPERIMENT = as_experiment_settings(
    {
        "name": "exp_kpp_exposure_ranking",
        "logic": {
            VARIATION1: KeyPreferredPartner,
        },
    }
)

CATBOOST_NO_FAST_SORT_EXPERIMENT = as_experiment_settings(
    {
        "name": "exp_catboost_remove_fast_sort",
        "logic": {
            VARIATION1: no_fast_sort_ranking.LogicPenaltyOnly,  # catboost + penalization
        },
    }
)


# Mapping from logic's nickname to logic's class
LOGICS_REGISTRY: dict[str, VendorLogicType] = {
    logic.NICKNAME: logic
    for logic in [
        price_parity.Logic,
        personalized_ranking.Logic,
        LogicTwoTowersV22,
        LogicTwoTowersV23,
        LogicTwoTowersV3,
        TTv2PriceParity,
        TTv23PriceParity,
        TTv3PriceParity,
        price_parity_v2.Logic,
    ]
}


def get_logics_registry():
    global LOGICS_REGISTRY
    return LOGICS_REGISTRY
