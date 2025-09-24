from typing import TYPE_CHECKING

import pytest

from abstract_ranking.input import FunWithFlags, Experiment
from vendor_ranking import personalized_ranking, price_parity, no_fast_sort_ranking
from vendor_ranking.configs.config import VendorRankingConfig, RankingLogicConfig
from vendor_ranking.price_parity.logic import TTv3PriceParity, TTv23PriceParity
from vendor_ranking.service import select_logic
from vendor_ranking.settings import get_logics_registry, VARIATION1
from vendor_ranking.two_tower.logic import LogicTwoTowersV22, LogicTwoTowersV23

if TYPE_CHECKING:
    from vendor_ranking.settings import VendorLogicType


@pytest.fixture(scope="session")
def logics_registry() -> dict[str, "VendorLogicType"]:
    return get_logics_registry()


@pytest.fixture(scope="session")
def ranking_config(vendor_ranking_config: VendorRankingConfig) -> RankingLogicConfig:
    return vendor_ranking_config.ranking


@pytest.mark.parametrize(
    ["experiment_name", "variation", "country_code", "expected_logic"],
    (
        # Holdout for any country
        ("exp_holdout_vl_23_q3", VARIATION1, None, personalized_ranking.Logic),
        # Experiments
        ("exp_price_parity_ranking", "Control", "BH", TTv3PriceParity),
        ("exp_price_parity_ranking", VARIATION1, "BH", price_parity.Logic),
        ("exp_catboost_remove_fast_sort", VARIATION1, "OM", no_fast_sort_ranking.LogicPenaltyOnly),
        ("exp_catboost_remove_fast_sort", VARIATION1, "AE", LogicTwoTowersV22),
        ("exp_UNKNOWN", "Control", None, LogicTwoTowersV22),  # unknown experiment name
        # Defaults
        (None, None, "BH", TTv3PriceParity),  # Bahrain
        (None, None, "OM", TTv3PriceParity),  # Oman
        (None, None, "UAE", LogicTwoTowersV22),  # UAE
        (None, None, "UNKNOWN", LogicTwoTowersV22),  # unknown country
        # Holdout experiments
        ("exp_ordering_and_food_finding_holdout_24_q1", VARIATION1, "AE", LogicTwoTowersV22),
        ("exp_ordering_and_food_finding_holdout_24_q1", VARIATION1, "BH", TTv23PriceParity),
        ("exp_ordering_and_food_finding_holdout_24_q1", VARIATION1, "EG", LogicTwoTowersV23),
        ("exp_holdout_vl_23_q3", VARIATION1, "AE", personalized_ranking.Logic),
        ("exp_holdout_vl_23_q3", VARIATION1, "BH", personalized_ranking.Logic),
    ),
)
def test_select_logic(
    logics_registry: dict[str, "VendorLogicType"],
    ranking_config: RankingLogicConfig,
    experiment_name: str,
    variation: str,
    country_code: str,
    expected_logic: "VendorLogicType",
):
    fwf = FunWithFlags(
        active_experiments=(experiment_name or {})
        and {experiment_name: Experiment(variation=variation)}
    )
    logic = select_logic(
        fwf, country_code, logics_registry=logics_registry, config=ranking_config
    )
    err_msg = f"[{country_code} {variation} {experiment_name}]{logic.NICKNAME}, {logic.MODEL_TAG}"
    assert logic is expected_logic, err_msg
