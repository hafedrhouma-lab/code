from pathlib import Path

import structlog
import yaml

LOG: "structlog.stdlib.BoundLogger" = structlog.getLogger(__name__)


def load_model_config(path: Path) -> dict:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    LOG.info(f"Loaded config from {path}")
    return config
