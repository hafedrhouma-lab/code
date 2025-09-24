import sys
from pathlib import Path
root_dir = Path(__file__).parent
[sys.path.insert(0, str(root_dir := root_dir.parent)) for _ in range(3)]
import os
import mlflow
import structlog
from dotenv import load_dotenv
from jinja2 import Template
from base.v1.db_utils import BigQuery
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import argparse
import pandas as pd
from google.cloud import bigquery
from google.api_core.exceptions import NotFound, BadRequest, Conflict
from google.api_core.exceptions import TooManyRequests
import time
import random
from datetime import datetime


os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv(override=True)
LOG: structlog.stdlib.BoundLogger = structlog.get_logger()
CURRENT_DIR = Path(__file__).parent
SQL_DIR = CURRENT_DIR / "sql"
LOG.info(f"Current directory: {CURRENT_DIR}")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

PROJECT_ID = os.getenv("PROJECT_ID")
# PROJECT_ID = 'tlb-data-dev'
DATASET_NAME = 'data_feature_store'



class ModelPredictor:
    def __init__(self, days_lag:int = 1):
        exp_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        self.MODEL_NAME = exp_name
        LOG.info(f"Tracking uri: {mlflow.get_tracking_uri()}")
        LOG.info(f"Experiment id: {exp_name}")

        self.output_table = f'{PROJECT_ID}.{DATASET_NAME}.item_ranking_content_based_account_embeddings'
        self.model = None
        self.bq_client = BigQuery()
        self.device = self.device_identifier()
        self.days_lag = days_lag
        self.output_schema = [
            bigquery.SchemaField("account_id", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("transformer_embeddings", "FLOAT64", mode="REPEATED"),
            bigquery.SchemaField("feature_timestamp", "TIMESTAMP", mode="REQUIRED"),
        ]

    @staticmethod
    def device_identifier():
        if torch.cuda.is_available():
            LOG.info("CUDA is available. Using GPU.")
            return 'cuda'
        else:
            LOG.info("CUDA is not available. Using CPU.")
            return 'cpu'

    def insert_guest_embeddings(self):
        '''runs the "guest_embeddings.sql'''
        sql_file = 'guest_embeddings.sql'
        with open(SQL_DIR / sql_file, 'r') as file:
            template = Template(file.read())

        query = template.render(
            output_table=self.output_table,
            days_lag=self.days_lag
        )
        self.bq_client.send_dml(query)


    def delete_today_records(self, batch_account_id):
        '''Deletes the records for the accounts in the list'''
        sql_file = 'delete_today_records.sql'
        with open(SQL_DIR / sql_file, 'r') as file:
            template = Template(file.read())

        account_list_str = ', '.join([str(account_id) for account_id in batch_account_id])
        query = template.render(
            account_id_list=account_list_str,
            output_table=self.output_table
        )
        retires = 0
        max_retries = 10
        while retires < max_retries:
            try:
                self.bq_client.send_dml(query)
                break
            except NotFound:
                LOG.warning("No table found to delete previous records")
                break
            except TooManyRequests:
                retires += 1
                random_wait = 20 + random.uniform(0, 180)  # Minimum 60 seconds + random 0-180 seconds
                LOG.warning(f"Retry {retires}: Waiting for {random_wait:.2f} seconds before retrying...")
                time.sleep(random_wait)
            except BadRequest as e:
                if "Could not serialize access to table" in str(e):
                    retires += 1
                    random_wait = 20 + random.uniform(0, 180)  # Minimum 60 seconds + random 0-180 seconds
                    LOG.warning(f"Retry {retires}: Waiting for {random_wait:.2f} seconds before retrying...")
                    time.sleep(random_wait)
                else:
                    LOG.error(e)
        if retires == max_retries:
            raise Exception("Exceeded maximum retries for BigQuery delete.")


    def __call__(self):

        LOG.info('Starting guest embeddings generation')
        batch = (-1, -2, -3, -4, -5, -6, -7, -8, -9, -10)
        self.delete_today_records(batch)
        LOG.info("Today's records deleted (Guest users)")
        self.insert_guest_embeddings()
        LOG.info("Guest embeddings updated successfully")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run ModelPredictor with specified parameters.")

    parser.add_argument("--days_lag", type=int, default=2, help="number of days to lag for active accounts")
    args = parser.parse_args()

    LOG.info(f"Arguments passed: {args}")

    run_predictions = ModelPredictor(
        days_lag=args.days_lag,
    )
    run_predictions()
