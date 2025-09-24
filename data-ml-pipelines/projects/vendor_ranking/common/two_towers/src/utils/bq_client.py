import os
import pickle
import yaml
from datetime import datetime

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


def write_to_bq(df, table_name, if_exists='append', schema=None):
    """
    Writes a Pandas DataFrame to a BigQuery table.

    Args:
        df (pd.DataFrame): The DataFrame to write to BigQuery.
        table_name (str): The destination table name in BigQuery.
        if_exists (str): The action to take if the table already exists.
                         Options are 'append', 'replace', and 'fail'.
                         Default is 'append'.
        schema: bq schema field to have proper casting
    """

    job_config = bigquery.LoadJobConfig(
        write_disposition={
            'append': bigquery.WriteDisposition.WRITE_APPEND,
            'replace': bigquery.WriteDisposition.WRITE_TRUNCATE,
            'fail': bigquery.WriteDisposition.WRITE_EMPTY
        }[if_exists],
        schema=schema
    )

    load_job = get_client().load_table_from_dataframe(df, table_name, job_config=job_config)

    load_job.result()
    LOG.info(f"Data has been written to {table_name}")


def create_clustered_partitioned_table(
        dataset_id, table_name, key_entity_field, key_entity_type, fields
):
    table_ref = get_client().dataset(dataset_id).table(table_name)

    try:
        get_client().get_table(table_ref)
        LOG.info(f"Table {table_name} already exists. Skipping creation.")
    except NotFound:
        type_map = {
            "integer": "INTEGER",
            "string": "STRING"
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

        schema.extend(
            bigquery.SchemaField(field, "STRING", mode="NULLABLE")
            for field in fields
            if field not in {key_entity_field, "feature_timestamp"}
        )

        table = bigquery.Table(table_ref, schema=schema)

        table.clustering_fields = [key_entity_field]
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="feature_timestamp"
        )

        table = get_client().create_table(table)
        LOG.info(f"{table_name} has been created")


def delete_data_for_date(
        table_name: str,
        timestamp: datetime.timestamp,
        country_code: str,
        date_column: str = "feature_timestamp"
):
    """
    Deletes data for a specific date and country code from the given table.

    Args:
        table_name (str): Name of the table to delete data from.
        timestamp (datetime.timestamp): Date to delete data for.
        country_code (str): Country code to filter the data.
        date_column (str): Name of the column containing the date. Defaults to "feature_timestamp".
    """
    today_date = timestamp.date().strftime('%Y-%m-%d')

    delete_query = f"""
        DELETE FROM `{table_name}`
        WHERE DATE({date_column}) = '{today_date}'
        AND country_code = '{country_code}'
    """

    query_job = get_client().query(delete_query)
    query_job.result()
    LOG.info(f"{today_date}'s Data has been Deleted from {table_name}")


def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the GCS bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    destination_path = destination_blob_name+source_file_name
    blob = bucket.blob(destination_path)
    blob.upload_from_filename(source_file_name)
    LOG.info(f"File {source_file_name} uploaded to {destination_blob_name}.")


def read_pkl_from_gcs(bucket_name, file_name, source_blob_name):
    """Reads a .pkl file from a GCS bucket and returns its content."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    destination_path = source_blob_name + file_name
    blob = bucket.blob(destination_path)
    pkl_data = blob.download_as_bytes()
    content = pickle.loads(pkl_data)

    LOG.info(f"File {file_name} read from {source_blob_name} in bucket {bucket_name}.")

    return content


def read_yaml_from_gcs(bucket_name, file_name, source_blob_name):
    """Reads a .yaml file from a GCS bucket and returns its content."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    destination_path = source_blob_name + file_name
    blob = bucket.blob(destination_path)
    yaml_data = blob.download_as_bytes()
    content = yaml.safe_load(yaml_data)
    LOG.info(f"File {file_name} read from {source_blob_name} in bucket {bucket_name}.")
    return content


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