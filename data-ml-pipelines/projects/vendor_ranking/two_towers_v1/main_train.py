import mlflow
from pathlib import Path
import os
import cloudpickle

from base.v0.file_tools import collect_folders
from base.v2.mlutils import register_and_assign_alias
from base.v2.talabat_pyfunc import talabat_log_model

from dotenv import load_dotenv
import structlog
import argparse

from projects.vendor_ranking.two_towers_v1.model_wrapper import ModelWrapper

from projects.vendor_ranking.common.two_towers.src.cli.train import train_model
from projects.vendor_ranking.common.two_towers.src.utils.load_utils import save_model_config
from projects.vendor_ranking.common.two_towers.src.utils.read_tf_records import read_tf_records
from projects.vendor_ranking.common.two_towers.src.utils.bq_client import (
    read_pkl_from_gcs,
    read_yaml_from_gcs
)

from projects.vendor_ranking.two_towers_v1 import (
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
    TFRECORDS_FILE_NAME,
    CHALLENGER_MODEL_ALIAS
)

from projects.vendor_ranking.common.two_towers.src.cli import (
    USER_MODEL_INPUT
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
# ENTRYPOINT ["python", "-m", "projects.vendor_ranking.two_towers_v1.main_training"]


class ModelTrainer:
    def __init__(self, country):
        exp_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        self.country = country
        self.MODEL_NAME = f"{exp_name}_{self.country.lower()}"

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
                model_info = talabat_log_model(
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
                    pip_requirements='requirements.txt'
                )

            register_and_assign_alias(
                model_info.model_uri,
                self.MODEL_NAME,
                CHALLENGER_MODEL_ALIAS
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
