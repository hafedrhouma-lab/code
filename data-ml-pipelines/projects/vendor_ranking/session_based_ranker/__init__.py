from pathlib import Path

BASE_PATH = Path(__file__).parent
PARAMS_PATH = Path(__file__).parent.resolve() / "model_config.yaml"

CONDA_ENV_PATH = BASE_PATH / "conda.yaml"

ARTIFACTS_PATH = './trained_model'

BUCKET_NAME = "tlb-data-dev-data-algorithms-content-optimization"
DESTINATION_BLOB_NAME = "session_based_model_training/data"
