import sys
from pathlib import Path
root_dir = Path(__file__).parent
[sys.path.insert(0, str(root_dir := root_dir.parent)) for _ in range(3)]
import os
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
from google.api_core.exceptions import TooManyRequests, Conflict, NotFound
import time
import random
import mlflow
from datetime import datetime


os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv(override=True)
LOG: structlog.stdlib.BoundLogger = structlog.get_logger()
CURRENT_DIR = Path(__file__).parent
SQL_DIR = CURRENT_DIR / "sql"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

PROJECT_ID = os.getenv("PROJECT_ID")
# PROJECT_ID = ('tlb-data-dev')
DATASET_NAME = 'data_feature_store'

class ModelPredictor:
    def __init__(self, country_code: str, batch_size: int = 100, float_precision: int = 16):
        self.output_table = f'{PROJECT_ID}.{DATASET_NAME}.item_ranking_content_based_item_embeddings'
        # self.output_table = f'tlb-data-prod.{DATASET_NAME}.item_ranking_content_based_item_embeddings'
        self.model = None
        self.bq_client = BigQuery()
        self.device = self.device_identifier()
        self.country_code = country_code
        self.batch_size = batch_size
        self.float_precision = float_precision
        self.output_schema = [
            bigquery.SchemaField("source_system_item_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("country_code", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("transformer_embeddings", "FLOAT64", mode="REPEATED"),
            bigquery.SchemaField("feature_timestamp", "TIMESTAMP", mode="NULLABLE"),
        ]
        self.run_name = f"item_{country_code}_{datetime.now().strftime('%Y-%m-%d')}"

    @staticmethod
    def device_identifier():
        if torch.cuda.is_available():
            LOG.info("CUDA is available. Using GPU.")
            return 'cuda'
        else:
            LOG.info("CUDA is not available. Using CPU.")
            return 'cpu'

    def generate_embeddings(self, documents):
        embeddings = self.model.encode(documents, normalize_embeddings=True)
        return embeddings.tolist()

    def generate_item_batches(self):
        """
        Generates batches of account IDs ending with a specific last digit.
        """
        sql_file ='get_new_items.sql'
        with open(SQL_DIR / sql_file, 'r') as file:
            template = Template(file.read())

        query = template.render(
            output_table=self.output_table,
            country_code=self.country_code
        )
        df = self.bq_client.read(query)
        LOG.info(f"New items found: {len(df)}")
        LOG.info(f"Query\n========\n{query}\n========")
        new_item_ids = df['source_system_item_id'].tolist()
        # split into lists of batch_size
        batch_list = [new_item_ids[i:i + self.batch_size] for i in range(0, len(new_item_ids), self.batch_size)]
        return batch_list

    def get_item_batch(self, batch_item_ids):
        '''Gets the users' previous orders and searches'''
        sql_file ='get_batch_item_token.sql'
        with open(SQL_DIR / sql_file, 'r') as file:
            template = Template(file.read())

        batch_item_ids_str = ', '.join([f'"{item_id}"' for item_id in batch_item_ids])

        query = template.render(
            batch_item_ids=batch_item_ids_str,
            country_code=self.country_code
        )
        df = self.bq_client.read(query)
        return df

    def create_table_if_not_exist(self):
        client = bigquery.Client()
        random_wait = random.uniform(0, 20)
        time.sleep(random_wait)
        try:
            table = client.get_table(self.output_table)
            LOG.warning(f"Table {table.table_id} exists.")
        except NotFound:
            LOG.warning("Table not found ... Creating one")
            table = bigquery.Table(self.output_table, schema=self.output_schema)
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="feature_timestamp",
            )
            table.clustering_fields = ["country_code"]
            table.description = "This table stores content-based item embeddings for all items"
            table.labels = {
               "table_owner": "fadi_baskharon",
               "dm_guardian": "na",
               "table_owner_slack_id": "u01rcdgp668",
               "dm_guardian_slack_id": "na",
                "du_channel_slack_id": "c085xsxmgsh",
                "time_grain": "daily",
                "dag_id": "item_ranking_content_based_generate_item_embeddings_inc"
            }
            # Create the table
            try:
                table = client.create_table(table)  # API request to create the table
                LOG.info(f"Created partitioned table {table.full_table_id}")
            except Conflict as e:
                LOG.warning(f"Table {self.output_table} already exists")

    def push_to_bq(self, df):

        max_retries = 10
        retries = 0
        while retries < max_retries:
            try:
                self.bq_client.write(df=df, table_name=self.output_table, if_exist_action='append', schema=self.output_schema)
                break
            except TooManyRequests:
                retries += 1
                random_wait = 60 + random.uniform(0, 180)  # Minimum 60 seconds + random 0-180 seconds
                LOG.warning(f"Retry {retries}: Waiting for {random_wait:.2f} seconds before retrying...")
                time.sleep(random_wait)
        if retries == max_retries:
            raise Exception("Exceeded maximum retries for BigQuery write.")

    def mlflow_logging(self):
        """fetch item_embeddings_stats.sql jinja2 template"""
        sql_file = "item_embeddings_stats.sql"
        with open(SQL_DIR / sql_file, 'r') as file:
            template = Template(file.read())
        query = template.render(
            output_table=self.output_table,
            country_code=self.country_code
        )
        df = self.bq_client.read(query)
        df['duplicates'] = df['count'] - df['count_distinct']
        metrics = df.to_dict(orient='records')[0]

        with mlflow.start_run(run_name=self.run_name):
            mlflow.log_metrics(metrics)
            mlflow.set_tags(
                {
                    'project_id': PROJECT_ID,
                    'table_name': self.output_table,
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'country_code': self.country_code,
                }
            )

    def __call__(self):

        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5', device=self.device)
        LOG.info(f"Model loaded: {self.model}")

        self.create_table_if_not_exist()

        batch_list = self.generate_item_batches()
        total_batches = len(batch_list)
        if total_batches == 0:
            LOG.info("No new items found.")
        else:
            LOG.info(f"Total batches: {total_batches} -- each of size: {len(batch_list[0])}")

        for i, batch in enumerate(batch_list):

            LOG.info(f"Processing batch: {i+1}/{total_batches}")
            LOG.info(f"New items found: {len(batch)}")
            df_batch = self.get_item_batch(batch)
            df_batch['country_code'] = self.country_code
            # Generate embeddings
            df_batch['transformer_embeddings'] = self.generate_embeddings(df_batch['tokens'].tolist())
            df_batch['transformer_embeddings'] = df_batch['transformer_embeddings'].apply(
                lambda x: [round(i, self.float_precision) for i in x])

            # Add time columns
            df_batch['feature_timestamp'] = pd.Timestamp('today').normalize()

            # Drop tokens
            df_batch.drop(columns=['chain_name', 'tokens'], inplace=True)

            # Push to BQ
            self.push_to_bq(df_batch)
            LOG.info(f"Batch of {len(df_batch)} items pushed to BQ.")

        self.mlflow_logging()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run ModelPredictor with specified parameters.")

    parser.add_argument("--country_code", type=str, default='KW', help="country code to process")
    parser.add_argument("--batch_size", type=int, default=1000, help="batch size for account_id processing")
    parser.add_argument("--float_precision", type=int, default=8, help="float precision for embeddings")

    args = parser.parse_args()

    LOG.info(f"Arguments passed: {args}")

    run_predictions = ModelPredictor(
        country_code=args.country_code,
        batch_size=args.batch_size,
        float_precision=args.float_precision

    )
    run_predictions()
