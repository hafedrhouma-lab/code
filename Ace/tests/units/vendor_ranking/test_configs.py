import pytest

from ace.configs.config import StageType
from vendor_ranking.configs.config import VendorRankingConfig, load_vendor_ranking_config


@pytest.mark.parametrize("stage", (StageType.TEST, StageType.QA, StageType.PROD))
def test_vendor_ranking_config_loading(stage: StageType):
    config: VendorRankingConfig = load_vendor_ranking_config(stage=stage)
    assert config
    assert isinstance(config, VendorRankingConfig)
    assert config.ranking and config.ranking.logic, "ranking logic settings must be present"
    assert config.ranking.logic.control.default, "no default logic for `Control` variation"
    assert config.ranking.logic.default.default, "no default logic for user with no experiments"
    assert config.ranking.logic.holdout, "no holdout experiments settings found"
