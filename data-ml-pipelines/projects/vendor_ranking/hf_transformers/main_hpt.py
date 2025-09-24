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

import argparse
import hypertune

load_dotenv(override=True)
LOG: structlog.stdlib.BoundLogger = structlog.get_logger()

from projects.vendor_ranking.hf_transformers.train import train_model
from projects.vendor_ranking.hf_transformers.model_wrapper import ModelWrapper
from projects.vendor_ranking.hf_transformers.online_model_wrapper import OnlineModelWrapper
from projects.vendor_ranking.hf_transformers import (
    PARAMS_PATH,
    ARTIFACTS_PATH,
    CONDA_ENV_PATH,
)
from projects.vendor_ranking.hf_transformers.utils.load_utils import load_model_config

from projects.vendor_ranking.hf_transformers.utils.data_utils import (
    read_data_gcs_fs
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--embedding_dimension',
        required=True,
        type=int,
        help='embedding dimension')
    parser.add_argument(
        '--epochs',
        required=True,
        type=int,
        help='epochs')
    parser.add_argument(
        '--country',
        required=True,
        type=str,
        help='country_code')

    parser.add_argument(
        '--dropout',
        required=True,
        type=float,
        help='dropout rate')
    parser.add_argument(
        '--learning_rate',
        required=True,
        type=float,
        help='learning rate')
    parser.add_argument(
        '--batch_size',
        required=True,
        type=int,
        help='batch size')
    parser.add_argument(
        '--gradient_accumulation_steps',
        required=True,
        type=int,
        help='gradient accumulation steps')
    parser.add_argument(
        '--weight_decay',
        required=True,
        type=float,
        help='weight decay')
    parser.add_argument(
        '--compression_dimension',
        required=True,
        type=int,
        help='compression dimension')

    args = parser.parse_args()
    return args

class TrainingParams(BaseSettings):
    train_date_start: str = Field(default="2024-07-18")
    train_date_end: str = Field(default="2024-09-08")
    test_date_start: str = Field(default="2024-09-09")
    test_date_end: str = Field(default="2024-09-15")

class ModelTrainer:
    def __init__(self):
        self.args = get_args()

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

            print(self.args.embedding_dimension)
            print(self.args.country)
            print(self.args.dropout)
            print(self.args.compression_dimension)

            # Fixed params
            self.params['area_id_dim'] = self.args.embedding_dimension
            self.params['geohash_dim'] = self.args.embedding_dimension
            self.params['order_hour_dim'] = self.args.embedding_dimension
            self.params['country_code'] = self.args.country
            self.params['training_args_gpu']['num_train_epochs'] = self.args.epochs
            self.params['training_args_cpu']['num_train_epochs'] = self.args.epochs

            # Tuning params
            self.params['dropout_rate'] = self.args.dropout
            self.params['t5_config_chain_id']['dropout_rate'] = self.args.dropout
            self.params['training_args_gpu']['learning_rate'] = self.args.learning_rate
            self.params['training_args_gpu']['per_device_train_batch_size'] = self.args.batch_size
            self.params['training_args_gpu']['gradient_accumulation_steps'] = self.args.gradient_accumulation_steps
            self.params['training_args_gpu']['weight_decay'] = self.args.weight_decay            
            self.params['compressed_dim'] = self.args.compression_dimension

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
                'chain_id_vocab': CHAIN_ID_VOCAB_PATH
            }

            LOG.info(f"Artifacts path: {ARTIFACTS_PATH}")
            LOG.info(f"Conda env path: {CONDA_ENV_PATH}")

            root_path = Path(__file__).parent.parent.parent.parent
            with collect_folders(
                    root=root_path,
                    paths=[
                        "projects/vendor_ranking/hf_transformers",
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

                hpt = hypertune.HyperTune()
                hpt.report_hyperparameter_tuning_metric(
                    hyperparameter_metric_tag='recall10',
                    metric_value=recall_10
                    # global_step=params.get("num_epochs")
                )


    def __call__(self):
        # the main logic of the class, where you call the methods in the needed order
        # self.preprocess_data()
        self.train()


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer()
