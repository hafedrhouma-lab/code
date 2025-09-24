from pathlib import Path

import structlog
import yaml

LOG: "structlog.stdlib.BoundLogger" = structlog.getLogger(__name__)


def load_model_config(path: Path) -> dict:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    LOG.info(f"Loaded config from {path}")
    return config


def save_model_config(file_path, params):
    with open(file_path, 'w') as file:
        yaml.dump(params, file, default_flow_style=False)