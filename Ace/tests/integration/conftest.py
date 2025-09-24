from typing import TYPE_CHECKING

import pytest

from ace.configs.config import StageType
from ace.configs.manager import ConfigManager

if TYPE_CHECKING:
    from ace.configs.config import AppConfig, AppS3Config


@pytest.fixture(scope="module")
def s3_app_config() -> "AppS3Config":
    config: "AppConfig" = ConfigManager.load_configuration(stage=StageType.TEST, as_dict=False)
    return config.storage.s3
