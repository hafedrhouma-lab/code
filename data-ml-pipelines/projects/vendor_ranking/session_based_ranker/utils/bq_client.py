import os
from datetime import datetime
import pandas as pd

import structlog
from google.cloud import bigquery, storage
from google.cloud.exceptions import NotFound

# project_id = "tlb-data-prod"
project_id = os.getenv("PROJECT_ID")

LOG: structlog.stdlib.BoundLogger = structlog.get_logger()


BQ_CLIENT: "bigquery.Client" = None


def get_client() -> "bigquery.Client":
    global BQ_CLIENT
    if BQ_CLIENT is None:
        BQ_CLIENT = bigquery.Client(project=project_id)
    return BQ_CLIENT


def read(query):
    df = get_client().query(
        query, project=project_id
    ).result()
    df = df.to_arrow(
        create_bqstorage_client=True,
        progress_bar_type="tqdm",
    ).to_pandas()
    return df


def write_to_bq(df, table_name, if_exists='append'):
    """
    Writes a Pandas DataFrame to a BigQuery table.

    Args:
        df (pd.DataFrame): The DataFrame to write to BigQuery.
        table_name (str): The destination table name in BigQuery.
        if_exists (str): The action to take if the table already exists.
                         Options are 'append', 'replace', and 'fail'.
                         Default is 'append'.
    """

    job_config = bigquery.LoadJobConfig(
        write_disposition={
            'append': bigquery.WriteDisposition.WRITE_APPEND,
            'replace': bigquery.WriteDisposition.WRITE_TRUNCATE,
            'fail': bigquery.WriteDisposition.WRITE_EMPTY
        }[if_exists],
    )

    load_job = get_client().load_table_from_dataframe(df, table_name, job_config=job_config)

    load_job.result()
    LOG.info(f"Data has been written to {table_name}")


def write_to_bq_batches(df, table_name, chunk_size=100000, if_exists='append'):
    """
    Writes a Pandas DataFrame to a BigQuery table in batches.

    Args:
        df (pd.DataFrame): The DataFrame to write to BigQuery.
        table_name (str): The destination table name in BigQuery.
        chunk_size (int): The number of rows per chunk to write to BigQuery.
        if_exists (str): The action to take if the table already exists.
                         Options are 'append', 'replace', and 'fail'.
                         Default is 'append'.
    """

    client = get_client()
    job_config = bigquery.LoadJobConfig(
        write_disposition={
            'append': bigquery.WriteDisposition.WRITE_APPEND,
            'replace': bigquery.WriteDisposition.WRITE_TRUNCATE,
            'fail': bigquery.WriteDisposition.WRITE_EMPTY
        }[if_exists],
    )

    for start in range(0, len(df), chunk_size):
        end = start + chunk_size
        chunk = df.iloc[start:end]

        load_job = client.load_table_from_dataframe(chunk, table_name, job_config=job_config)
        try:
            load_job.result()  # Wait for the job to complete.
            LOG.info(f"Chunk {start} to {end} has been written to {table_name}")
        except Exception as e:
            LOG.error(f"Failed to write chunk {start} to {end} to BigQuery: {e}")
            raise


def create_clustered_partitioned_table(
        dataset_id, table_name, key_entity_field, key_entity_type, fields
):
    client = get_client()
    table_ref = client.dataset(dataset_id).table(table_name)

    try:
        table = client.get_table(table_ref)
        LOG.info(f"Table {table_name} already exists.")

    except NotFound:
        LOG.info(f"Table {table_name} does not exist. Creating a new table...")
        type_map = {
            "integer": "INTEGER",
            "string": "STRING",
            "float_array": "FLOAT64"  # array handling via REPEATED mode

        }

        key_entity_field_type = type_map.get(key_entity_type.lower())
        if key_entity_field_type is None:
            raise ValueError(
                "key_entity_type must be either 'integer' or 'string'"
            )

        schema = [
            bigquery.SchemaField(key_entity_field, key_entity_field_type, mode="REQUIRED"),
            bigquery.SchemaField("feature_timestamp", "TIMESTAMP", mode="REQUIRED")
        ]

        # Add other fields to the schema, using REPEATED mode for arrays
        for field, field_type in fields.items():
            if field not in {key_entity_field, "feature_timestamp"}:
                field_type_mapped = type_map.get(field_type.lower(), "STRING")
                field_mode = "REPEATED" if field_type == "float_array" else "NULLABLE"
                schema.append(bigquery.SchemaField(field, field_type_mapped, mode=field_mode))

        table = bigquery.Table(table_ref, schema=schema)

        table.clustering_fields = [key_entity_field]
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="feature_timestamp"
        )

        table = client.create_table(table)
        LOG.info(f"{table_name} has been created")


def delete_data_for_date(table_name: str, timestamp: datetime.timestamp, country_code):
    today_date = timestamp.date().strftime('%Y-%m-%d')

    delete_query = f"""
        DELETE FROM `{table_name}`
        WHERE DATE(feature_timestamp) = '{today_date}'
        AND country_code = '{country_code}'
    """

    query_job = get_client().query(delete_query)
    query_job.result()
    LOG.info(f"{today_date}'s Data has been Deleted from {table_name}")


def delete_data_for_country(table_name: str, country_code: str):
    delete_query = f"""
        DELETE FROM `{table_name}`
        WHERE country_code = '{country_code}'
    """

    query_job = get_client().query(delete_query)
    query_job.result()
    LOG.info(f"Country: {country_code}'s Data has been Deleted from {table_name}")


def merge_tables(staging_table_name, main_table_name, key_columns, update_columns=[]):
    """
    Merges data from the staging table into the main table in BigQuery based on key columns.
    Args:
        staging_table_name (str): The name of the staging table.
        main_table_name (str): The name of the main target table.
        dataset_id (str): The dataset ID.
        key_columns (list): List of columns to merge on.
        update_columns (list): List of additional columns to update other than key columns.
    """
    client = get_client()

    update_columns_string = [f'T.{col} = S.{col}' for col in update_columns]
    update_set_statement = ", ".join(update_columns_string)

    merge_statement = f"""
    MERGE `{main_table_name}` AS T
    USING `{staging_table_name}` AS S
    ON {' AND '.join([f'T.{col} = S.{col}' for col in key_columns])}
    WHEN MATCHED THEN
        UPDATE SET {update_set_statement}
    WHEN NOT MATCHED THEN
        INSERT ({', '.join(key_columns + update_columns)})
        VALUES ({', '.join(['S.' + col for col in key_columns + update_columns])})
    """

    LOG.info(f"Executing MERGE operation from {staging_table_name} to {main_table_name}...")
    query_job = client.query(merge_statement)
    try:
        query_job.result()  # Wait for the job to complete.
        LOG.info(f"Data successfully merged into {main_table_name}")
    except Exception as e:
        LOG.error(f"Failed to merge data into BigQuery: {e}")
        raise


def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    destination_path = destination_blob_name+source_file_name
    blob = bucket.blob(destination_path)
    blob.upload_from_filename(source_file_name)
    LOG.info(f"File {source_file_name} uploaded to {destination_blob_name}.")


def check_if_blob_exists(bucket_name, file_name, source_blob_name):
    """Check if a blob (file) exists in a GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    source_path = source_blob_name + file_name
    blob = bucket.blob(source_path)

    return blob.exists()


def read_parquet_from_gcs(bucket_name, file_name, source_blob_name, local_path='/tmp'):
    """Downloads a .parquet file from a GCS bucket and returns its content."""

    # Initialize GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    source_path = source_blob_name + file_name
    blob = bucket.blob(source_path)

    local_file_path = os.path.join(local_path, file_name)
    blob.download_to_filename(local_file_path)
    data_df = pd.read_parquet(local_file_path)
    os.remove(local_file_path)

    LOG.info(f"File {file_name} downloaded from {source_blob_name} in bucket {bucket_name} and read as parquet.")

    return data_df


def update_timestamp_column(date: str, table_name: str, country_code):
    """
    Update the `feature_timestamp` column in the specified BigQuery table,
    setting the time to '00:00:00 UTC' for all rows matching the provided date.

    Parameters:
        date (str): The date in 'YYYY-MM-DD' format to filter rows for updating.
        table_name (str): The full table name in the format `project_id.dataset_id.table_id`.

    Example usage:
        update_timestamp_column('2024-09-04', 'my_project.my_dataset.my_table')
    """
    try:
        datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Date must be in 'YYYY-MM-DD' format")

    update_query = f"""
        UPDATE `{table_name}`
        SET feature_timestamp = TIMESTAMP('{date} 00:00:00 UTC')
        WHERE DATE(feature_timestamp) = '{date}'
        AND country_code = '{country_code}'
    """

    query_job = get_client().query(update_query)
    query_job.result()

    LOG.info(f"Rows with date {date} have been updated in table {table_name}.")
