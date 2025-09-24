import sys
import json
from pathlib import Path
import mlflow
import os
from mlflow.models.model import ModelInfo
from base.v0.file_tools import collect_folders
from pydantic import Field
from pydantic_settings import BaseSettings

root_dir = Path(__file__).parent
[sys.path.insert(0, str(root_dir := root_dir.parent)) for _ in range(3)]

import structlog
from dotenv import load_dotenv

load_dotenv(override=True)
LOG: structlog.stdlib.BoundLogger = structlog.get_logger()

from projects.vendor_ranking.session_based_ranker.train import train_model
from projects.vendor_ranking.session_based_ranker.model_wrapper import ModelWrapper
from projects.vendor_ranking.session_based_ranker.online_model_wrapper import OnlineModelWrapper
from projects.vendor_ranking.session_based_ranker import (
    PARAMS_PATH,
    ARTIFACTS_PATH,
    CONDA_ENV_PATH,
)
from projects.vendor_ranking.session_based_ranker.utils.load_utils import load_model_config

from projects.vendor_ranking.session_based_ranker.utils.data_utils import (
    read_data_fs,
    read_data_gcs_fs
)


class TrainingParams(BaseSettings):
    train_date_start: str = Field(default="2024-10-17")
    train_date_end: str = Field(default="2024-11-17")
    test_date_start: str = Field(default="2024-11-18")
    test_date_end: str = Field(default="2024-11-24")

class ModelTrainer:
    def __init__(self):
        exp_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        self.MODEL_NAME = exp_name
        LOG.info(f"Tracking uri: {mlflow.get_tracking_uri()}")
        LOG.info(f"Experiment id: {exp_name}")

    def get_data(self):
        self.data = read_data_gcs_fs(
            self.param_dates.train_date_start,
            self.param_dates.train_date_end,
            self.params["country_code"],
            self.params["data_points"]
        )

    def preprocess_data(self):
        # Data preprocessing code
        pass

    def train(self):
        with mlflow.start_run() as run:  # Do not change this line

            self.param_dates = TrainingParams()
            self.params = load_model_config(PARAMS_PATH)
            LOG.info(f"Params : {self.params}")

            mlflow.set_tag("country", self.params['country_code'])
            self.get_data()

            for key, value in self.params.items():
                mlflow.log_param(key, value)

            model, fast_tokenizer, recall_10 = train_model(
                self.params,
                self.param_dates,
                self.data
            )
                
            # Save the model and tokenizer
            model.save_models(ARTIFACTS_PATH)
            fast_tokenizer.save_pretrained(ARTIFACTS_PATH)

            # Save the training configs
            TRAINING_CONFIG_PATH = os.path.join(ARTIFACTS_PATH, "training_config.json")
            with open(TRAINING_CONFIG_PATH, 'w') as f:
                json.dump(self.params, f)

            # Define paths for the generated artifacts
            OFFLINE_MODEL_PATH = os.path.join(ARTIFACTS_PATH, "offline_model", "offline_pytorch_model.bin")
            ONLINE_MODEL_PATH = os.path.join(ARTIFACTS_PATH, "online_model", "online_pytorch_model.bin")
            CONFIG_PATH = os.path.join(ARTIFACTS_PATH, "model_config.json")
            TOKENIZER_PATH = os.path.join(ARTIFACTS_PATH, "tokenizers")
            AREA_ID_MAPPING_PATH = os.path.join(ARTIFACTS_PATH, "area_id_to_index.json")
            GEOHASH_MAPPING_PATH = os.path.join(ARTIFACTS_PATH, "geohash_to_index.json")
            CHAIN_ID_VOCAB_PATH = os.path.join(ARTIFACTS_PATH, "chain_id_vocab.json")

            # Collect artifacts
            artifacts = {
                'offline_model': OFFLINE_MODEL_PATH,
                'online_model': ONLINE_MODEL_PATH,
                'model_config': CONFIG_PATH,
                'training_config': TRAINING_CONFIG_PATH,
                'tokenizers': TOKENIZER_PATH,
                'area_id_to_index': AREA_ID_MAPPING_PATH,
                'geohash_to_index': GEOHASH_MAPPING_PATH,
                'chain_id_vocab': CHAIN_ID_VOCAB_PATH
            }

            # Collect artifacts for the online model
            artifacts_online = {
                'online_model': ONLINE_MODEL_PATH,
                'model_config': CONFIG_PATH,
                'training_config': TRAINING_CONFIG_PATH,
                'area_id_to_index': AREA_ID_MAPPING_PATH,
                'geohash_to_index': GEOHASH_MAPPING_PATH,
                'chain_id_vocab': CHAIN_ID_VOCAB_PATH,
                'tokenizers': TOKENIZER_PATH
            }

            LOG.info(f"Artifacts path: {ARTIFACTS_PATH}")
            LOG.info(f"Conda env path: {CONDA_ENV_PATH}")

            root_path = Path(__file__).parent.parent.parent.parent
            with collect_folders(
                    root=root_path,
                    paths=[
                        "projects/vendor_ranking/session_based_ranker",
                        "projects/vendor_ranking/common",
                        "base",
                    ]
            ) as tmp_dir:
                code_path = [
                    tmp_dir / "projects",
                    tmp_dir / "base"
                ]

                # Log the entire all 3 models
                model_info_offline: "ModelInfo" = mlflow.pyfunc.log_model(
                    artifact_path="src/full_model",
                    python_model=ModelWrapper(),
                    artifacts=artifacts,
                    conda_env=str(CONDA_ENV_PATH),
                    code_path=code_path
                )
                LOG.info(
                    f"OFFLINE Model: `{self.MODEL_NAME}` logged to mlflow server. "
                    f"Run URI: {model_info_offline.model_uri}. "
                    f"Params {self.params}."
                )

                # Log only the online model
                model_info_online: "ModelInfo" = mlflow.pyfunc.log_model(
                    artifact_path="src/model",
                    python_model=OnlineModelWrapper(),
                    artifacts=artifacts_online,
                    conda_env=str(CONDA_ENV_PATH),
                    code_path=code_path
                )
                LOG.info(
                    f"ONLINE Model: `{self.MODEL_NAME}` logged to mlflow server. "
                    f"Run URI: {model_info_online.model_uri}. "
                    f"Params {self.params}."
                )


    def __call__(self):
        # the main logic of the class, where you call the methods in the needed order
        # self.preprocess_data()
        self.train()


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer()
