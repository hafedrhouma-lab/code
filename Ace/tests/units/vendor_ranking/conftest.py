import pytest

from ace.configs.config import StageType
from vendor_ranking.configs.config import VendorRankingConfig, load_vendor_ranking_config


@pytest.fixture(scope="session")
def vendor_ranking_config() -> VendorRankingConfig:
    return load_vendor_ranking_config(StageType.TEST)
