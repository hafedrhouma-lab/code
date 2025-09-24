import os
import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import cloudpickle
import mlflow
import structlog
from dotenv import load_dotenv
from mlflow.models.model import ModelInfo
from pydantic import Field
from pydantic_settings import BaseSettings

from base.v0.file_tools import collect_folders
from projects.vendor_ranking.common.two_towers.src.cli.prepare.feast.prepare_param import PrepareData
from projects.vendor_ranking.common.two_towers.src.cli.prepare.feast.prepare_train_df import TrainingDatasetGenerator
from projects.vendor_ranking.common.two_towers.src.cli.prepare.feast.prepare_utils import PrepareUtils
from projects.vendor_ranking.common.two_towers.src.cli.prepare.prepare_tf_records import PrepareTFRecords
from projects.vendor_ranking.common.two_towers.src.cli.train import train_model
from projects.vendor_ranking.common.two_towers.src.utils.bq_client import upload_to_gcs
from projects.vendor_ranking.common.two_towers.src.utils.load_utils import load_model_config
from projects.vendor_ranking.common.two_towers.src.utils.read_tf_records import read_tf_records
from projects.vendor_ranking.two_towers_v1.model_wrapper import ModelWrapper
from . import (
    USER_WEIGHTS_PATH,
    CHAIN_WEIGHTS_PATH,
    PARAMS_INPUT_DATA_PATH,
    MODEL_CONFIG_PATH,
    BASE_PATH,
    CONDA_ENV_PATH,
    PARAMS_PATH,
    ESE_CHAIN_EMBEDDINGS_PATH,
    BUCKET_NAME,
    TFRECORDS_FILE_NAME,
    DESTINATION_BLOB_NAME,
    FEATURE_DESCRIPTION_FILE
)

if TYPE_CHECKING:
    from mlflow import ActiveRun
    
load_dotenv(override=True)
LOG: "structlog.stdlib.BoundLogger" = structlog.getLogger(__name__)


class TrainingParams(BaseSettings):
    train_date_start: str = Field(default="2023-10-31", env="TRAIN_DATE_START")
    train_date_end: str = Field(default="2023-11-26", env="TRAIN_DATE_END")
    test_date_start: str = Field(default="2023-11-27", env="TEST_DATE_START")
    test_date_end: str = Field(default="2023-11-28", env="TEST_DATE_END")


class ModelTrainer:
    def __init__(self, country):
        exp_name = os.environ["MLFLOW_EXPERIMENT_NAME"]
        self.country = country
        self.MODEL_NAME = f"{exp_name}_{self.country.lower()}"
        LOG.info(f"Tracking uri: {mlflow.get_tracking_uri()}")
        LOG.info(f"Experiment id: {exp_name}")
        LOG.info(f"MODEL_NAME: {self.MODEL_NAME}")

    def run_training_pipeline(self):
        with mlflow.start_run() as self.run:  # type: ActiveRun
            params_dates = TrainingParams()
            params = load_model_config(PARAMS_PATH)

            params_input_data = PrepareData(
                train_date_start=params_dates.train_date_start,
                train_date_end=params_dates.train_date_end,
                test_date_start=params_dates.test_date_start,
                test_date_end=params_dates.test_date_end,
                country_code=params.get("country"),
                query_features=params.get("query_features"),
                candidate_features=params.get("candidate_features")
            )
            data_training_utils = PrepareUtils(
                params_dates.train_date_start,
                params_dates.train_date_end,
                params.get("country"),
                params.get("EMBEDDING_DATE")
            )

            train_data_generator = TrainingDatasetGenerator(
                train_date_start=params_dates.train_date_start,
                train_date_end=params_dates.train_date_end,
                params=params
            )
            train_ds = train_data_generator.generate()

            tfrecords_preparer = PrepareTFRecords(
                train_ds,
                train_data_generator.positive_data.shape[0],
                train_data_generator.negative_data.shape[0],
                params,
                FEATURE_DESCRIPTION_FILE,
                TFRECORDS_FILE_NAME
            )
            tfrecords_preparer.generate()
            upload_to_gcs(BUCKET_NAME, TFRECORDS_FILE_NAME, DESTINATION_BLOB_NAME)

            train_ds = read_tf_records(
                "feature_description.pkl",
                f"gs://{BUCKET_NAME}/{DESTINATION_BLOB_NAME}"
            )

            model, recall_values = train_model(
                train_ds,
                params,
                params_input_data,
                data_training_utils
            )

            # Dataset Metadata to MLFlow
            positive_sample = train_data_generator.positive_data.sample(
                n=10, random_state=42
            )
            negative_sample = train_data_generator.negative_data.sample(
                n=10,
                random_state=42
            )
            mlfow_dataset = mlflow.data.from_pandas(
                positive_sample,
                name=f"<pos><{params_dates.train_date_start}><{params_dates.train_date_end}>"
            )
            mlflow.log_input(
                mlfow_dataset,
                context="training"
            )
            mlfow_dataset = mlflow.data.from_pandas(
                negative_sample,
                name=f"<neg><{params_dates.train_date_start}><{params_dates.train_date_end}>"
            )
            mlflow.log_input(
                mlfow_dataset,
                context="training"
            )

            # Artefact 1: User/Chain model weights
            model.customer_model.save_weights(
                f"{USER_WEIGHTS_PATH}/user_model_weights",
                save_format="tf"
            )
            model.chain_model.save_weights(
                f"{CHAIN_WEIGHTS_PATH}/chain_model_weights",
                save_format="tf"
            )
            # Artefact 2: Ese Chain Embeddings
            data_training_utils.ese_chain_embeddings.to_parquet(
                ESE_CHAIN_EMBEDDINGS_PATH,
                index=True,
                engine="pyarrow"
            )
            # Artefact 3: Params input data
            with open(PARAMS_INPUT_DATA_PATH, "wb") as f:
                cloudpickle.dump(params_input_data, f)
            with open("data_training_utils.pkl", "wb") as f:
                cloudpickle.dump(data_training_utils, f)

            # Artefact 4: Model config
            for key, value in params.items():
                mlflow.log_param(key, value)

            LOG.info(f"Conda env path: {CONDA_ENV_PATH}")
            root_path: Path = BASE_PATH.parent.parent.parent

            with collect_folders(
                    root=root_path,
                    paths=[
                        "projects/vendor_ranking/two_towers_v1_eg",
                        "projects/vendor_ranking/common",
                        "base",
                    ]
            ) as tmp_dir:

                model_info: "ModelInfo" = mlflow.pyfunc.log_model(
                    artifact_path="src/model",
                    python_model=ModelWrapper(),
                    artifacts={
                        "user_weights": USER_WEIGHTS_PATH,
                        "chain_weights": CHAIN_WEIGHTS_PATH,
                        "ese_chain_embeddings": ESE_CHAIN_EMBEDDINGS_PATH,
                        "params_input": PARAMS_INPUT_DATA_PATH,
                        "model_config": MODEL_CONFIG_PATH,
                    },
                    code_path=[
                        tmp_dir / "projects",
                        tmp_dir / "base"
                    ],
                    conda_env=str(CONDA_ENV_PATH)
                )

        recall10 = recall_values["test_recall@10"][-1]
        LOG.info(
            f"Model: `{self.MODEL_NAME}` logged to mlflow server. "
            f"Run URI: {model_info.model_uri}. "
            f"Params {recall10=}."
        )

    def __call__(self):
        self.run_training_pipeline()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train two tower model.")
    parser.add_argument(
        "--country",
        required=True,
        choices=['EG', 'AE', 'KW', 'QA', 'OM', 'BH', 'IQ', 'JO'],
        help="Possible country values are: ['EG', 'AE', 'KW', 'QA', 'OM', 'BH', 'IQ', 'JO']"
    )
    args = parser.parse_args()

    training_pipeline = ModelTrainer(country=args.country)
    training_pipeline()