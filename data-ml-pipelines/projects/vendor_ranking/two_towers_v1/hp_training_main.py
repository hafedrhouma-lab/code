import mlflow
from pathlib import Path
import os
import cloudpickle
from base.v0.file_tools import collect_folders
from dotenv import load_dotenv
import structlog

import argparse
import hypertune

from projects.vendor_ranking.two_towers_v1.model_wrapper import ModelWrapper

from projects.vendor_ranking.common.two_towers.src.cli.train import train_model
from projects.vendor_ranking.common.two_towers.src.utils.load_utils import save_model_config
from projects.vendor_ranking.common.two_towers.src.utils.read_tf_records import read_tf_records
from projects.vendor_ranking.common.two_towers.src.utils.bq_client import (
    read_pkl_from_gcs,
    read_yaml_from_gcs
)

from . import (
    USER_WEIGHTS_PATH,
    CHAIN_WEIGHTS_PATH,
    PARAMS_INPUT_DATA_PATH,
    DATA_TRAINING_UTILS,
    MODEL_CONFIG_PATH,
    BASE_PATH,
    CONDA_ENV_PATH,
    ESE_CHAIN_EMBEDDINGS_PATH,
    BUCKET_NAME,
    DESTINATION_BLOB_NAME,
    FEATURE_DESCRIPTION_FILE,
    TFRECORDS_FILE_NAME
)

load_dotenv(override=True)
LOG: "structlog.stdlib.BoundLogger" = structlog.getLogger(__name__)


# FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/tf2-cu121.2-15.py310
#
# WORKDIR /app
# COPY . /app
#
# ENV GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json
# ENV MLFLOW_TRACKING_URI=https://data.talabat.com/api/public/mlflow
# ENV GOOGLE_CLOUD_PROJECT=tlb-data-dev
#
#
# RUN pip install cloudml-hypertune
# RUN pip install --no-cache-dir -r requirements.txt
#
#
# ENTRYPOINT ["python", "-m", "projects.vendor_ranking.two_towers_v1.hp_training_main"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--temperature',
        required=True,
        type=float,
        help='temperature')
    parser.add_argument(
        '--embedding_dimension',
        required=True,
        type=int,
        help='embedding dimension')
    parser.add_argument(
        '--num_epochs',
        required=True,
        type=int,
        help='number of epochs')
    parser.add_argument(
        '--country',
        required=True,
        type=str,
        help='country_code')
    args = parser.parse_args()
    return args


class ModelTrainer:
    def __init__(self):
        self.args = get_args()

        exp_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        self.country = self.args.country
        self.MODEL_NAME = f"{exp_name}_{self.country}"

        LOG.info(f"Tracking uri: {mlflow.get_tracking_uri()}")
        LOG.info(f"Experiment id: {exp_name}")

    def run_training_pipeline(self):
        with mlflow.start_run() as self.run:
            mlflow.set_tag("country", self.country)

            source_blob = f"{DESTINATION_BLOB_NAME}/{self.country}/"

            params = read_yaml_from_gcs(
                BUCKET_NAME,
                f'model_config_{self.country}.yaml',
                source_blob
            )
            params['temperature'] = self.args.temperature
            params['embedding_dimension'] = self.args.embedding_dimension
            params['num_epochs'] = self.args.num_epochs
            save_model_config(MODEL_CONFIG_PATH, params)

            for key, value in params.items():
                mlflow.log_param(key, value)

            params_input_data = read_pkl_from_gcs(
                BUCKET_NAME,
                PARAMS_INPUT_DATA_PATH,
                source_blob
            )
            data_training_utils = read_pkl_from_gcs(
                BUCKET_NAME,
                DATA_TRAINING_UTILS,
                source_blob
            )
            feature_description = read_pkl_from_gcs(
                BUCKET_NAME,
                FEATURE_DESCRIPTION_FILE,
                source_blob
            )

            train_ds = read_tf_records(
                feature_description,
                f'gs://{BUCKET_NAME}/{source_blob}{TFRECORDS_FILE_NAME}'
            )
            model, recall_values = train_model(
                train_ds,
                params,
                params_input_data,
                data_training_utils
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

            LOG.info(f"Conda env path: {CONDA_ENV_PATH}")
            root_path: Path = BASE_PATH.parent.parent.parent

            with collect_folders(
                    root=root_path,
                    paths=[
                        "projects/vendor_ranking/two_towers_v1",
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

        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='recall10',
            metric_value=recall10,
            global_step=params.get("num_epochs"))

    def __call__(self):
        self.run_training_pipeline()


if __name__ == "__main__":
    training_pipeline = ModelTrainer()
    training_pipeline()
