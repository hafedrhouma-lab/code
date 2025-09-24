import structlog
import argparse
from base.v1 import mlutils
import mlflow
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import warnings
from dotenv import load_dotenv
import torch
import time
from torch.utils.data import DataLoader

from projects.vendor_ranking.hf_transformers.utils.preprocess import PreprocessDataset
from projects.vendor_ranking.hf_transformers.utils.data_utils import (
    read_data_inference_fs,
    read_data_inference_gf_fs,
    read_data_active_inference_fs
)
from projects.vendor_ranking.hf_transformers.utils.bq_client import (
    write_to_bq,
    create_clustered_partitioned_table,
    delete_data_for_country,
    delete_data_for_date,
    update_timestamp_column
)


warnings.simplefilter(action='ignore', category=FutureWarning)
load_dotenv(override=True)
LOG: structlog.stdlib.BoundLogger = structlog.get_logger()


DATASET_ID = "data_feature_store"
TABLE_NAME = "account_embeddings_t5_all_source"
DST_TABLE_NAME = f"{DATASET_ID}.{TABLE_NAME}"
KEY_ENTITY_FIELD = "account_id"
KEY_ENTITY_TYPE = "integer"
FIELDS = {
    "account_id": "integer",
    "embeddings": "float_array",
    "feature_timestamp": "timestamp",
    "country_code": "string",
}

CURRENT_DATE_MINUS_ONE = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
#CURRENT_DATE_MINUS_ONE = '2024-10-31'
DATE_TO_WRITE = datetime.now() - timedelta(days=1)

BATCH_SIZE = 10000
WRITE_CHUNK_SIZE = 100000



class OfflineEmbeddings:
    def __init__(self, country, initial_load):
        exp_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        self.country = country
        self.initial_load = initial_load
        self.MODEL_NAME = f"{exp_name}_full_model_{self.country.lower()}"
        LOG.info(f"Tracking uri: {mlflow.get_tracking_uri()}")
        LOG.info(f"Experiment id: {exp_name}")
        LOG.info(f"MODEL_NAME: {self.MODEL_NAME}")

    @staticmethod
    def collate_fn(batch):
        batch_features = {key: torch.stack([sample[key] for sample in batch]) for key in batch[0]}
        return batch_features

    def __call__(self, start_date=None):

        create_clustered_partitioned_table(
            DATASET_ID, TABLE_NAME, KEY_ENTITY_FIELD, KEY_ENTITY_TYPE, FIELDS
        )

        LOG.info("Retrieving the model from mlflow server...")
        model_dict = mlutils.load_registered_model(
            model_name=self.MODEL_NAME,
            alias='best_full_model',
        )
        mlflow_model = model_dict['mlflow_model']
        model = mlflow_model.unwrap_python_model()

        offline_model = model.offline_model
        model_config = model.model_config


        LOG.info("Loading data for inference ...")

        if self.initial_load:
            data_df = read_data_inference_fs(start_date=CURRENT_DATE_MINUS_ONE, end_date=CURRENT_DATE_MINUS_ONE, country_code=self.country)
        else:
            data_df = read_data_active_inference_fs(start_date=CURRENT_DATE_MINUS_ONE, end_date=CURRENT_DATE_MINUS_ONE, country_code=self.country)

        data_gf_df = read_data_inference_gf_fs(start_date=CURRENT_DATE_MINUS_ONE, end_date=CURRENT_DATE_MINUS_ONE, country_code=self.country)
        # data_gf_df = read_data_inference_gf_fs(start_date='2024-10-24', end_date='2024-10-24', country_code=self.country)

        data_df = pd.concat([data_gf_df, data_df], ignore_index=True)
        LOG.info(f"Total accounts for inference: {data_df.shape[0]} in country: {self.country}")

        account_ids = data_df['account_id'].tolist()

        data_dict = data_df.to_dict(orient='records')

        dataset = PreprocessDataset(
            data_dict,
            model_config["feature_configs"],
            model_config["numerical_features"]
        )
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=self.collate_fn)

        start_time = time.time()

        if self.initial_load:
            delete_data_for_country(DST_TABLE_NAME, country_code=self.country)
        else:
            delete_data_for_date(DST_TABLE_NAME, DATE_TO_WRITE, country_code=self.country)


        with torch.no_grad():
            for idx, batch_samples in enumerate(dataloader):
                offline_output = offline_model(**batch_samples)

                # Prepare batch DataFrame for BigQuery upload
                batch_embeddings = offline_output.numpy().astype(np.float32).tolist()
                batch_account_ids = account_ids[idx * BATCH_SIZE: (idx + 1) * BATCH_SIZE]

                output_df = pd.DataFrame({
                    'account_id': batch_account_ids,
                    'embeddings': batch_embeddings
                })
                output_df["feature_timestamp"] = DATE_TO_WRITE
                output_df["country_code"] = self.country

                # Write each batch to BigQuery
                LOG.info(f"Writing Chunk {idx * BATCH_SIZE} to {((idx + 1) * BATCH_SIZE) - 1} to {DST_TABLE_NAME}")

                write_to_bq(df=output_df, table_name=DST_TABLE_NAME, if_exists="append")


        end_time = time.time()
        offline_time = end_time - start_time
        LOG.info(f"Offline model inference time: {offline_time:.2f} seconds")

        update_timestamp_column(
            DATE_TO_WRITE.strftime('%Y-%m-%d'),
            DST_TABLE_NAME,
            self.country
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Account embeddings generation for T5 model."
    )
    parser.add_argument(
        "--country",
        required=True,
        choices=['EG', 'AE', 'KW', 'QA', 'OM', 'BH', 'IQ', 'JO'],
        help="Possible country values are: ['EG', 'AE', 'KW', 'QA', 'OM', 'BH', 'IQ', 'JO']"
    )
    parser.add_argument(
        "--initial_load",
        required=True,
        choices=['True', 'False'],
        help="Set to True for an initial load or False otherwise."
    )
    args = parser.parse_args()
    initial_load = args.initial_load == 'True'
    offline_embeddings = OfflineEmbeddings(country=args.country, initial_load=initial_load)
    offline_embeddings()
