import sys
from pathlib import Path
root_dir = Path(__file__).parent
[sys.path.insert(0, str(root_dir := root_dir.parent)) for _ in range(3)]
import torch
import os
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count()-1)
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count()-1)
os.environ["NUMEXPR_NUM_THREADS"] = str(os.cpu_count()-1)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
TORCH_NUM_THREADS = os.cpu_count()-1
# Configure PyTorch for optimal performance
torch.set_float32_matmul_precision('high')

torch.set_num_threads(TORCH_NUM_THREADS)
torch.set_num_interop_threads(TORCH_NUM_THREADS)
import mlflow
import structlog
from dotenv import load_dotenv
from jinja2 import Template
from base.v1.db_utils import BigQuery
from sentence_transformers import SentenceTransformer
import argparse
import pandas as pd
from google.cloud import bigquery
from google.api_core.exceptions import NotFound, BadRequest, Conflict
from google.api_core.exceptions import TooManyRequests
import time
import random
from datetime import datetime


load_dotenv(override=True)
LOG: structlog.stdlib.BoundLogger = structlog.get_logger()
CURRENT_DIR = Path(__file__).parent
SQL_DIR = CURRENT_DIR / "sql"
LOG.info(f"Current directory: {CURRENT_DIR}")


PROJECT_ID = os.getenv("PROJECT_ID")
# PROJECT_ID = 'tlb-data-dev'
DATASET_NAME = 'data_feature_store'



class ModelPredictor:
    def __init__(self, last_digit: int, days_lag:int = 1, batch_size: int = 100, max_order_rank: int = 15, float_precision: int = 16):
        exp_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        self.MODEL_NAME = exp_name
        LOG.info(f"Tracking uri: {mlflow.get_tracking_uri()}")
        LOG.info(f"Experiment id: {exp_name}")
        self.order_table = 'tlb-data-prod.data_platform.ese_orders_token'
        self.search_table = 'tlb-data-prod.data_platform.ese_keywords_token'
        self.output_table = f'{PROJECT_ID}.{DATASET_NAME}.item_ranking_content_based_account_embeddings'
        self.model = None
        self.bq_client = BigQuery()
        self.device = self.device_identifier()
        self.last_digit = last_digit
        self.days_lag = days_lag
        self.batch_size = batch_size
        self.max_order_rank = max_order_rank
        self.float_precision = float_precision
        self.output_schema = [
            bigquery.SchemaField("account_id", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("transformer_embeddings", "FLOAT64", mode="REPEATED"),
            bigquery.SchemaField("feature_timestamp", "TIMESTAMP", mode="REQUIRED"),
        ]
        self.run_name = f"account_last_digit_{last_digit}_{datetime.now().strftime('%Y-%m-%d')}"

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

    def generate_account_batches(self):
        """
        Generates batches of account IDs ending with a specific last digit.
        """
        sql_file ='get_new_accounts.sql'
        with open(SQL_DIR / sql_file, 'r') as file:
            template = Template(file.read())

        query = template.render(
            last_digit=self.last_digit,
            days_lag=self.days_lag
        )
        df = self.bq_client.read(query)
        account_ids = df['account_id'].tolist()
        # split into lists of batch_size
        batch_list = [account_ids[i:i + self.batch_size] for i in range(0, len(account_ids), self.batch_size)]
        return batch_list

    def get_batch_account_history(self, batch_account_ids):
        '''Gets the accounts' previous orders and searches'''
        sql_file = 'get_batch_account_token.sql'
        with open(SQL_DIR / sql_file, 'r') as file:
            template = Template(file.read())

        batch_account_ids_str = ', '.join([str(account_id) for account_id in batch_account_ids])
        query = template.render(
            batch_account_ids=batch_account_ids_str,
            order_table=self.order_table,
            search_table=self.search_table,
            max_order_rank=self.max_order_rank,
        )
        df = self.bq_client.read(query)
        df['tokens'] = df['tokens'].apply(lambda x: x.replace(' + ', ', '))
        # df['tokens'] = df['tokens'].str.replace(' + ', ', ') # better not to use apply
        return df

    def copy_last_day_records(self):
        '''Copies the records from the previous day to the current day'''
        sql_file = 'copy_last_day_records.sql'
        with open(SQL_DIR / sql_file, 'r') as file:
            template = Template(file.read())

        query = template.render(
            last_digit=self.last_digit,
            output_table=self.output_table
        )
        try:
            self.bq_client.send_dml(query)
        except NotFound:
            LOG.warning("Table not found to copy previous records")
            random_wait = random.uniform(0, 20)
            time.sleep(random_wait)
            self.create_table()

    def create_table(self):
        client = bigquery.Client()
        table = bigquery.Table(self.output_table, schema=self.output_schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="feature_timestamp",
        )
        table.description = "This table stores content-based account embeddings for all registered accounts"
        table.labels = {
            "table_owner": "fadi_baskharon",
            "dm_guardian": "na",
            "table_owner_slack_id": "u01rcdgp668",
            "dm_guardian_slack_id": "na",
            "du_channel_slack_id": "c085xsxmgsh",
            "time_grain": "daily",
            "dag_id": "item_ranking_content_based_generate_account_embeddings_inc"
        }
        try:
            table = client.create_table(table)
            LOG.info(f"Created partitioned table {table.full_table_id}")
        except Conflict as e:
            LOG.warning(f"Table {self.output_table} already exists")

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
        """fetch account_embeddings_stats.sql jinja2 template"""
        sql_file = "account_embeddings_stats.sql"
        with open(SQL_DIR / sql_file, 'r') as file:
            template = Template(file.read())
        query = template.render(
            last_digit=self.last_digit,
            output_table=self.output_table,
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
                    'last_digit': self.last_digit,
                    'date': datetime.now().strftime('%Y-%m-%d')
                }
            )

    def __call__(self):

        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5', device=self.device, backend="onnx")
        LOG.info(f"Model loaded: {self.model}")

        # self.copy_last_day_records()  # FS handles historical data

        batch_list = self.generate_account_batches()
        total_batches = len(batch_list)
        LOG.info(f"Total batches: {total_batches} -- each of size: {self.batch_size}")

        for i, batch in enumerate(batch_list):

            LOG.info(f"Processing batch: {i+1}/{total_batches}")
            df_batch = self.get_batch_account_history(batch)
            LOG.info(f"Batch of size {len(df_batch)} accounts fetched.")

            # Generate embeddings
            LOG.info("Generating embeddings for the batch")
            df_batch['transformer_embeddings'] = self.generate_embeddings(df_batch['tokens'].tolist())
            df_batch['transformer_embeddings'] = df_batch['transformer_embeddings'].apply(
                lambda x: [round(i, self.float_precision) for i in x])
            LOG.info("Embeddings generated.")

            # Add time columns
            df_batch['feature_timestamp'] = pd.Timestamp('today').normalize()

            # Drop tokens
            df_batch.drop(columns=['tokens'], inplace=True)

            # Delete today's records if exist
            self.delete_today_records(batch)

            # Push to BQ
            LOG.info("Pushing to BQ")
            self.push_to_bq(df_batch)
            LOG.info(f" Batch of size {len(df_batch)} accounts pushed to BQ.")

        self.mlflow_logging()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run ModelPredictor with specified parameters.")

    parser.add_argument("--last_digit", type=int, default=0, help="last account_id digit to process")
    parser.add_argument("--days_lag", type=int, default=1, help="number of days to lag for active accounts")
    parser.add_argument("--batch_size", type=int, default=3000, help="batch size for account_id processing")
    parser.add_argument("--max_order_rank", type=int, default=15, help="max lookback rank for orders and searches. max = 50")
    parser.add_argument("--float_precision", type=int, default=8, help="float precision for embeddings")

    args = parser.parse_args()

    LOG.info(f"Arguments passed: {args}")

    run_predictions = ModelPredictor(
        last_digit=args.last_digit,
        days_lag=args.days_lag,
        batch_size=args.batch_size,
        max_order_rank=args.max_order_rank,
        float_precision=args.float_precision

    )
    run_predictions()
