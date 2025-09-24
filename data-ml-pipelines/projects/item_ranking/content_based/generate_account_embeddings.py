import sys
from pathlib import Path
root_dir = Path(__file__).parent
[sys.path.insert(0, str(root_dir := root_dir.parent)) for _ in range(3)]

import os
# Calculate resources based on container limits
# Container has 10 CPUs and 32GB RAM, but we'll use 8 for compute
NUM_CPUS = 10
COMPUTE_CPUS = 8  # Leave 2 cores for I/O and system tasks
MEMORY_GB = 32

# Configure OpenMP/MKL threads
os.environ["OMP_NUM_THREADS"] = str(COMPUTE_CPUS)
os.environ["MKL_NUM_THREADS"] = str(COMPUTE_CPUS)
os.environ["NUMEXPR_NUM_THREADS"] = str(COMPUTE_CPUS)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Configure PyTorch
import torch
torch.set_num_threads(COMPUTE_CPUS)  # Use 8 threads for compute
torch.set_num_interop_threads(2)  # Keep inter-op threads low
torch.set_float32_matmul_precision('high')

# Set memory limits only for CPU since we're not using CUDA
MAX_MEMORY = int((MEMORY_GB - 4) * 1024 * 1024 * 1024)  # Convert to bytes
if hasattr(torch, 'set_memory_limit'):
    try:
        torch.set_memory_limit(MAX_MEMORY)
    except Exception as e:
        LOG.warning(f"Could not set memory limit: {e}")

import mlflow
import structlog
import asyncio
from dotenv import load_dotenv
from google.cloud import storage
from google.cloud import bigquery
import pandas as pd
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from datetime import datetime
import concurrent.futures
from google.cloud.storage.blob import Blob
from google.api_core import retry
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from google.api_core.exceptions import Conflict

# Configure environment
load_dotenv(override=True, interpolate=True)
LOG: structlog.stdlib.BoundLogger = structlog.get_logger()
CURRENT_DIR = Path(__file__).parent
SQL_DIR = CURRENT_DIR / "sql"
LOG.info(f"Current directory: {CURRENT_DIR}")

class EmbeddingGenerator:
    def __init__(self, 
                 project_id: str, 
                 bucket_name: str,
                 bucket_path: str,
                 model_name: str = 'BAAI/bge-small-en-v1.5',
                 batch_size: int = 10000,
                 max_workers: int = 4,
                 last_digit: int = 0):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.last_digit = last_digit
        
        # Initialize clients
        self.storage_client = storage.Client()
        self.bq_client = bigquery.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        self.bucket_path = bucket_path
        
        # Initialize paths - include bucket_path in the paths
        self.input_path = f"{bucket_path}/embedding_pipeline/digit_{last_digit}/input"
        self.output_path = f"{bucket_path}/embedding_pipeline/digit_{last_digit}/output"
        
        # Align workers with CPU config
        self.max_workers = min(max_workers, COMPUTE_CPUS)
        
        # Initialize model for CPU only since we're not using CUDA
        self.device = 'cpu'  # Force CPU usage
        
        # Initialize model with basic config first
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        
        # Set model to evaluation mode to optimize memory usage
        self.model.eval()
        
        # Configure model dtype after initialization
        for param in self.model.parameters():
            param.data = param.data.to(torch.float32)  # Use FP32 for CPU
        
        # Disable gradients since we're only doing inference
        torch.set_grad_enabled(False)
        
        # Generate a unique run ID using timestamp and random hex
        self.run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
        LOG.info(f"Generated unique run ID: {self.run_id}")
        
        LOG.info(f"Initialized with model: {model_name} on device: {self.device}")
        LOG.info(f"Using input path: {self.input_path}")
        LOG.info(f"Using output path: {self.output_path}")
        
        # Define schema for BigQuery table
        self.output_schema = [
            bigquery.SchemaField("account_id", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("transformer_embeddings", "FLOAT64", mode="REPEATED"),
            bigquery.SchemaField("feature_timestamp", "TIMESTAMP", mode="REQUIRED")
        ]

    def export_to_gcs(self, query: str) -> str:
        """Export BigQuery data to GCS in Parquet format with parallel uploads"""
        LOG.info("Exporting data from BigQuery to GCS...")
        
        # Use run_id in paths
        base_path = f"{self.input_path}/{self.run_id}"
        
        # Create temporary table with the query results and partitioning column
        temp_table = f"{self.project_id}.data_feature_store.account_embeddings_export_{self.run_id}"
        create_temp_table = f"""
        CREATE OR REPLACE TABLE {temp_table} AS 
        WITH source_data AS ({query}),
        partitioned_data AS (
            SELECT 
                *,
                MOD(ABS(FARM_FINGERPRINT(CAST(account_id AS STRING))), {self.batch_size}) as partition_id
            FROM source_data
        )
        SELECT * FROM partitioned_data
        """
        LOG.info(f"Creating temporary table: {temp_table}")
        self.bq_client.query(create_temp_table).result()
        
        try:
            # Get number of partitions
            partition_query = f"""
            SELECT COUNT(DISTINCT partition_id) as num_partitions 
            FROM {temp_table}
            """
            num_chunks = self.bq_client.query(partition_query).result().to_dataframe().iloc[0]['num_partitions']
            LOG.info(f"Data will be split into {num_chunks} chunks")
            
            async def upload_chunk(chunk_idx: int):
                chunk_table = f"{temp_table}_chunk_{chunk_idx}"
                try:
                    # Create chunk table
                    chunk_query = f"""
                    CREATE OR REPLACE TABLE {chunk_table} AS
                    SELECT account_id, tokens 
                    FROM {temp_table}
                    WHERE partition_id = {chunk_idx}
                    """
                    
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.bq_client.query(chunk_query).result()
                    )
                    
                    # Configure export
                    job_config = bigquery.ExtractJobConfig()
                    job_config.destination_format = bigquery.DestinationFormat.PARQUET
                    job_config.compression = bigquery.Compression.SNAPPY
                    
                    destination_uri = f"gs://{self.bucket_name}/{base_path}/chunk-{chunk_idx:05d}.parquet"
                    
                    # Extract to GCS
                    extract_job = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.bq_client.extract_table(
                            bigquery.TableReference.from_string(chunk_table),
                            destination_uri,
                            job_config=job_config,
                            location='US'
                        )
                    )
                    
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        extract_job.result
                    )
                    
                    LOG.info(f"Exported chunk {chunk_idx + 1}/{num_chunks}")
                    return destination_uri
                    
                finally:
                    # Clean up chunk table
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.bq_client.delete_table(chunk_table, not_found_ok=True)
                    )
            
            # Run all uploads in parallel
            async def run_parallel_uploads():
                tasks = [upload_chunk(chunk_idx) for chunk_idx in range(num_chunks)]
                return await asyncio.gather(*tasks, return_exceptions=True)
            
            # Run the async upload process
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            results = loop.run_until_complete(run_parallel_uploads())
            
            # Check for failures
            failures = [r for r in results if isinstance(r, Exception)]
            if failures:
                LOG.error(f"Failed to export {len(failures)} chunks")
                raise failures[0]
            
            # Verify uploads
            blobs = list(self.bucket.list_blobs(prefix=base_path))
            LOG.info(f"Export completed successfully. Created {len(blobs)} files")
            
            return base_path
            
        finally:
            # Cleanup temporary table
            LOG.info(f"Cleaning up temporary table: {temp_table}")
            self.bq_client.delete_table(temp_table, not_found_ok=True)

    async def process_chunk(self, blob: Blob) -> Dict:
        """Process a single chunk of data"""
        local_path = f"/tmp/{blob.name.split('/')[-1]}"
        result_local_path = f"/tmp/result_{blob.name.split('/')[-1]}"
        
        try:
            # Download and process chunk
            await asyncio.get_event_loop().run_in_executor(
                None, blob.download_to_filename, local_path
            )
            
            # Read chunk in batches to reduce memory usage
            chunk_df = pd.read_parquet(local_path)
            LOG.info(f"Processing chunk with {len(chunk_df)} rows")
            
            # Process in smaller sub-batches
            sub_batch_size = 100  # Reduced from 500
            embeddings_list = []
            
            for i in range(0, len(chunk_df), sub_batch_size):
                sub_batch = chunk_df['tokens'].iloc[i:i + sub_batch_size].tolist()
                
                # Generate embeddings
                sub_embeddings = self.model.encode(
                    sub_batch,
                    normalize_embeddings=True,
                    batch_size=32,
                    show_progress_bar=False
                )
                
                embeddings_list.extend([[float(val) for val in emb] for emb in sub_embeddings])
                
                # Clear memory after each sub-batch
                del sub_batch, sub_embeddings
                import gc
                gc.collect()
                
                LOG.info(f"Processed sub-batch {i//sub_batch_size + 1}/{(len(chunk_df) + sub_batch_size - 1)//sub_batch_size}")
            
            # Create result DataFrame efficiently
            result_df = pd.DataFrame({
                'account_id': chunk_df['account_id'].astype(np.int64),
                'transformer_embeddings': embeddings_list,
                'feature_timestamp': pd.Timestamp.now(tz='UTC').floor('D')
            })
            
            # Clear original chunk_df
            del chunk_df
            gc.collect()
            
            # Create PyArrow table with explicit schema
            table = pa.Table.from_pydict(
                {
                    'account_id': pa.array(result_df['account_id'], type=pa.int64()),
                    'transformer_embeddings': pa.array(result_df['transformer_embeddings'], type=pa.list_(pa.float64())),
                    'feature_timestamp': pa.array(result_df['feature_timestamp'], type=pa.timestamp('us'))
                }
            )
            
            # Write to parquet
            pq.write_table(
                table,
                result_local_path,
                compression='snappy',
                use_dictionary=False,
                write_statistics=True
            )
            
            # Upload to GCS
            output_blob_name = blob.name.replace(self.input_path, self.output_path)
            output_blob = self.bucket.blob(output_blob_name)
            
            await asyncio.get_event_loop().run_in_executor(
                None, output_blob.upload_from_filename, result_local_path
            )
            
            return {'input_blob': blob.name, 'output_blob': output_blob_name}
            
        finally:
            # Cleanup
            if os.path.exists(local_path):
                os.remove(local_path)
            if os.path.exists(result_local_path):
                os.remove(result_local_path)

    async def process_chunks(self, input_path: str) -> List[str]:
        """Process all chunks concurrently"""
        blobs = list(self.bucket.list_blobs(prefix=input_path))
        LOG.info(f"Processing {len(blobs)} chunks...")
        
        # Process chunks concurrently with semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_with_semaphore(blob):
            async with semaphore:
                try:
                    # Add timeout to prevent indefinite hanging
                    result = await asyncio.wait_for(
                        self.process_chunk(blob),
                        timeout=3600  # 1 hour timeout per chunk
                    )
                    LOG.info(f"Successfully processed chunk: {blob.name}")
                    return result
                except asyncio.TimeoutError:
                    LOG.error(f"Timeout processing chunk {blob.name}")
                    raise
                except Exception as e:
                    LOG.error(f"Error processing chunk {blob.name}: {e}")
                    raise
        
        tasks = [process_with_semaphore(blob) for blob in blobs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out any failed results
        successful_results = [r for r in results if isinstance(r, dict)]
        if len(successful_results) != len(results):
            LOG.warning(f"Failed to process {len(results) - len(successful_results)} chunks")
        
        return [result['output_blob'] for result in successful_results]

    async def _download_and_read_parquet(self, blob: Blob) -> pd.DataFrame:
        """Download and read a single parquet file asynchronously"""
        local_path = f"/tmp/load_{blob.name.split('/')[-1]}"
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, blob.download_to_filename, local_path
            )
            return pd.read_parquet(local_path)
        finally:
            if os.path.exists(local_path):
                os.remove(local_path)

    def create_table(self, destination_table: str):
        """Create BigQuery table if it doesn't exist"""
        LOG.info(f"Creating table if not exists: {destination_table}")
        client = bigquery.Client()
        table_id = f"{self.project_id}.{destination_table}"
        
        table = bigquery.Table(table_id, schema=self.output_schema)
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
        except Conflict:
            LOG.info(f"Table {table_id} already exists, skipping creation")
        except Exception as e:
            LOG.error(f"Error creating table: {e}")
            raise

    async def load_bigquery_via_pandas(self, output_path: str, destination_table: str):
        """Load processed results to BigQuery through pandas DataFrame"""
        LOG.info(f"Loading results to BigQuery: {destination_table}")
        
        # Ensure table exists
        self.create_table(destination_table)
        
        # List all parquet files in the output path
        blobs = list(self.bucket.list_blobs(prefix=output_path))
        if not blobs:
            raise ValueError(f"No parquet files found in {output_path}")
        
        table_ref = f"{self.project_id}.{destination_table}"
        table = self.bq_client.get_table(table_ref)
        
        # Process files in smaller batches to avoid memory issues
        file_batch_size = 2  # Reduce from 5 to 2 files at a time
        total_rows = 0
        
        for i in range(0, len(blobs), file_batch_size):
            batch_blobs = blobs[i:i + file_batch_size]
            
            # Read batch of parquet files
            tasks = [self._download_and_read_parquet(blob) for blob in batch_blobs]
            dfs = await asyncio.gather(*tasks)
            
            # Combine batch DataFrames
            batch_df = pd.concat(dfs, ignore_index=True)
            LOG.info(f"Processing file batch {i//file_batch_size + 1}, rows: {len(batch_df)}")
            
            # Process the combined DataFrame in smaller chunks for BigQuery insertion
            rows_per_chunk = 1000  # Smaller chunks for BigQuery insertion
            for start_idx in range(0, len(batch_df), rows_per_chunk):
                end_idx = min(start_idx + rows_per_chunk, len(batch_df))
                chunk_df = batch_df.iloc[start_idx:end_idx]
                
                # Convert chunk to records and insert
                chunk_records = chunk_df.to_dict('records')
                errors = self.bq_client.insert_rows(
                    table, 
                    chunk_records,
                    self.output_schema
                )
                if errors:
                    raise Exception(f"Encountered errors while inserting rows: {errors}")
                
                total_rows += len(chunk_records)
                LOG.info(f"Inserted {len(chunk_records)} rows, total: {total_rows}")
                
                # Clear chunk memory
                del chunk_records, chunk_df
            
            # Clear batch memory
            del batch_df, dfs
            import gc
            gc.collect()
        
        LOG.info(f"Successfully loaded {total_rows} total rows to {destination_table}")

    async def cleanup_gcs_folders(self, input_path: str, output_path: str):
        """Delete input and output folders from GCS after successful loading"""
        LOG.info("Cleaning up GCS folders...")
        try:
            # Delete input folder blobs
            input_blobs = self.bucket.list_blobs(prefix=input_path)
            for blob in input_blobs:
                blob.delete()
            LOG.info(f"Deleted input folder: {input_path}")

            # Delete output folder blobs
            output_blobs = self.bucket.list_blobs(prefix=output_path)
            for blob in output_blobs:
                blob.delete()
            LOG.info(f"Deleted output folder: {output_path}")

        except Exception as e:
            LOG.error(f"Error cleaning up GCS folders: {e}")
            raise

    def run_pipeline(self, query: str, destination_table: str, input_path: str = None):
        """Run the complete pipeline"""
        try:
            if input_path:
                LOG.info(f"Using existing data from GCS: {input_path}")
            else:
                input_path = self.export_to_gcs(query)
                LOG.info(f"Data exported to GCS: {input_path}")
            
            # Process chunks concurrently
            output_paths = asyncio.run(self.process_chunks(input_path))
            LOG.info(f"Processed {len(output_paths)} chunks")
            
            # Load results to BigQuery using pandas
            if output_paths:
                output_dir = os.path.dirname(output_paths[0])
                asyncio.run(self.load_bigquery_via_pandas(output_dir, destination_table))
                LOG.info("Results loaded to BigQuery via pandas")
                
                # Clean up GCS folders after successful loading
                asyncio.run(self.cleanup_gcs_folders(input_path, output_dir))
                LOG.info("GCS folders cleaned up successfully")
            else:
                raise Exception("No chunks were successfully processed")
            
            LOG.info("Pipeline completed successfully")
            
        except Exception as e:
            LOG.error(f"Pipeline failed: {e}")
            raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run EmbeddingGenerator with specified parameters.")
    
    parser.add_argument("--last_digit", type=int, default=0, help="last account_id digit to process")
    parser.add_argument("--days_lag", type=int, default=1, help="number of days to lag for active accounts")
    parser.add_argument("--batch_size", type=int, default=3000, help="batch size for account_id processing")
    parser.add_argument("--max_order_rank", type=int, default=15, help="max lookback rank for orders and searches. max = 50")
    parser.add_argument("--max_workers", type=int, default=4, help="number of concurrent workers")
    
    args = parser.parse_args()
    LOG.info(f"Arguments passed: {args}")

    PROJECT_ID = os.getenv("PROJECT_ID")
    DATASET_NAME = os.getenv("DATASET_NAME", "data_feature_store")
    BUCKET_NAME = os.getenv("GCS_BUCKET_NAME").format(PROJECT_ID=PROJECT_ID)
    BUCKET_PATH = os.getenv("GCS_BUCKET_PATH")

    if not all([PROJECT_ID, BUCKET_NAME]):
        raise ValueError("Missing required environment variables. Please check PROJECT_ID and BUCKET_NAME in .env file")

    LOG.info(f"Using Project ID: {PROJECT_ID}")
    LOG.info(f"Using Dataset: {DATASET_NAME}")
    LOG.info(f"Using Bucket: {BUCKET_NAME}")
    LOG.info(f"Using Bucket Path: {BUCKET_PATH}")
    # Initialize generator
    generator = EmbeddingGenerator(
        project_id=PROJECT_ID,
        bucket_name=BUCKET_NAME,
        bucket_path=BUCKET_PATH,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        last_digit=args.last_digit
    )
    
    # Construct query using the same structure as get_batch_account_token.sql
    query = f"""
    WITH orders AS (
        SELECT 
            account_id, 
            chain_name, 
            item_name, 
            item_description, 
            order_rank
        FROM `tlb-data-prod.data_platform.ese_orders_token`
        WHERE 1=1
            AND DATE(order_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL {args.days_lag} DAY)
            AND MOD(account_id, 10) = {args.last_digit}
            AND order_rank <= {args.max_order_rank}
        ORDER BY order_time DESC
    ),
    keywords AS (
        SELECT 
            account_id, 
            search_history, 
            search_rank
        FROM `tlb-data-prod.data_platform.ese_keywords_token`
        WHERE 1=1
            AND DATE(search_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL {args.days_lag} DAY)
            AND MOD(account_id, 10) = {args.last_digit}
            AND search_rank <= {args.max_order_rank}
        ORDER BY search_time DESC
    )
    -- Tokenization on the fly
    SELECT
        COALESCE(orders.account_id, keywords.account_id) AS account_id,
        ARRAY_TO_STRING(
            [
                STRING_AGG(
                    CONCAT(orders.chain_name, ' ', orders.item_name, ' ', orders.item_description),
                    ' '
                    ORDER BY orders.order_rank
                ),
                STRING_AGG(
                    keywords.search_history,
                    ' '
                    ORDER BY keywords.search_rank
                )
            ], ' '
        ) AS tokens
    FROM
        orders
        FULL OUTER JOIN keywords
        ON orders.account_id = keywords.account_id
        AND orders.order_rank = keywords.search_rank
    GROUP BY account_id

    """
    
    
    destination_table = f"{DATASET_NAME}.item_ranking_content_based_account_embeddings"
    
    generator.run_pipeline(
        query=query,
        destination_table=destination_table,
    ) 