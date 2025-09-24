import yaml
import structlog
from pathlib import Path
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

from projects.vendor_ranking.common.two_towers.src.utils.bq_client import get_client

LOG: structlog.stdlib.BoundLogger = structlog.get_logger()

BASE_PATH = Path(__file__).parent
MONITORING_TABLE_PATH: str = str((BASE_PATH / "monitoring_tables.yaml").absolute())

DATASET_ID = "data_playground"
with open(MONITORING_TABLE_PATH, 'r') as file:
    TABLES_CONFIG = yaml.safe_load(file)


def create_table(dataset_id, table_name, fields):
    """Create a BigQuery table if it doesn't exist."""

    client = get_client()

    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_name)
    schema = [bigquery.SchemaField(field['name'], field['type']) for field in fields]

    try:
        client.get_table(table_ref)
        LOG.info(f"Table {dataset_id}.{table_name} already exists.")
    except NotFound:
        table = bigquery.Table(table_ref, schema=schema)
        client.create_table(table)
        LOG.info(f"Table {dataset_id}.{table_name} created.")


if __name__ == '__main__':
    for table in TABLES_CONFIG['monitoring_tables']:
        create_table(DATASET_ID, table['table_name'], table['fields'])

