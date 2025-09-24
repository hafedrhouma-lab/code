#from tutils.db_utils import BigQuery
from google.cloud import bigquery
#from tutils.db_utils import BigQuery

def date_to_str(date_str):
    return date_str.replace("-", "_")


def read_query(query):
    #bq_client = BigQuery()
    bq_client = bigquery.Client()
    df = bq_client.query(query).result()
    df = df.to_arrow(
        create_bqstorage_client=True,
        progress_bar_type="tqdm",
    ).to_pandas()
    return df


def get_project():
    bq_client = bigquery.Client()
    return bq_client.project