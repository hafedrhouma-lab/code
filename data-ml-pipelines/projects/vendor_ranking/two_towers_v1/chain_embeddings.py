import os
import argparse
import warnings
from datetime import datetime, timedelta
import mlflow
import numpy as np
import pandas as pd
import structlog
from dotenv import load_dotenv

from projects.vendor_ranking.common.two_towers.src.data import (
    query_loader,
    get_data_fetcher,
    get_feature_store
)

from projects.vendor_ranking.common.two_towers.src.data.processors.feast import (
    ChainFeaturesProcessor
)
from projects.vendor_ranking.common.two_towers.src.utils.bq_client import (
    write_to_bq,
    create_clustered_partitioned_table,
    delete_data_for_date,
    update_timestamp_column
)
from projects.vendor_ranking.common.two_towers.src.cli.prepare import (
    CHAIN_FEATURE_COLUMNS
)
from projects.vendor_ranking.common.two_towers.evaluation import (
    load_mlflow_model
)
from projects.vendor_ranking.two_towers_v1 import (
    PROD_MODEL_ALIAS
)

warnings.simplefilter(action='ignore', category=FutureWarning)
load_dotenv(override=True)
LOG: structlog.stdlib.BoundLogger = structlog.get_logger()

DST_TABLE_NAME = f"data_feature_store.chain_embeddings_two_tower_mlf_all"
DATASET_ID = "data_feature_store"
TABLE_NAME = "chain_embeddings_two_tower_mlf_all"
KEY_ENTITY_FIELD = "chain_id"
KEY_ENTITY_TYPE = "integer"
FIELDS = ["chain_id", "feature_timestamp", "embeddings", "country_code"]

CURRENT_DATE_MINUS_ONE = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
CURRENT_DATE_MINUS_THIRTY = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
DATE_TO_WRITE = datetime.now() - timedelta(days=1)


# CURRENT_DATE_MINUS_ONE = '2024-09-08'
# CURRENT_DATE_MINUS_THIRTY = '2024-09-01'
# DATE_TO_WRITE = datetime.now() - timedelta(days=1)


class ChainEmbeddings:
    def __init__(self, country):
        self.exp_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        self.country = country
        self.MODEL_NAME = f"{self.exp_name}_{self.country.lower()}"
        LOG.info(f"Tracking uri: {mlflow.get_tracking_uri()}")
        LOG.info(f"Experiment id: {self.exp_name}")
        LOG.info(f"MODEL_NAME: {self.MODEL_NAME}")
        self.model_input = {}

    def __call__(self, start_date=None):
        create_clustered_partitioned_table(
            DATASET_ID, TABLE_NAME, KEY_ENTITY_FIELD, KEY_ENTITY_TYPE, FIELDS
        )

        LOG.info("Retrieving the model from mlflow server...")
        model = load_mlflow_model(
            self.MODEL_NAME,
            PROD_MODEL_ALIAS
        )
        self.params = model.two_tower_model.params

        candidate_features = self.params.get("candidate_features")

        LOG.info("Start Querying Chain features...")
        chain_features_query = query_loader.load_query(
            'chain_features.sql.j2',
            start_date=CURRENT_DATE_MINUS_THIRTY,
            end_date=CURRENT_DATE_MINUS_ONE,
            country_code=self.params.get("country"),
            orders_profile_table=get_feature_store().get_data_source("orders_profile").get_table_query_string(),
            chain_profile_table=get_feature_store().get_data_source("chain_profile").get_table_query_string()
        )
        df_chain_features = get_data_fetcher().fetch_data(
            description='chain features',
            source="feast",
            entity_sql=chain_features_query,
            features=CHAIN_FEATURE_COLUMNS
        )
        chain_features_processor = ChainFeaturesProcessor(
            df_chain_features,
            self.params.get('candidate_features')
        )
        chain_features = chain_features_processor.process()

        LOG.info("Predicting using the model...")
        chain_embeddings = model.two_tower_model.chain_model(
            chain_features[candidate_features]
        ).numpy()

        assert np.isnan(chain_embeddings).sum() == 0

        LOG.info("Prepare DataFrame to store embeddings...")
        chain_ids = chain_features["chain_id"].tolist()
        chain_embeddings_df = pd.DataFrame(
            {"chain_id": chain_ids, "embeddings": chain_embeddings.tolist()}
        )
        chain_embeddings_df['chain_id'] = chain_embeddings_df['chain_id'].astype('int')
        chain_embeddings_df['embeddings'] = chain_embeddings_df['embeddings'].astype('str')
        chain_embeddings_df.embeddings = chain_embeddings_df.embeddings.apply(
            lambda x: x.replace("[", "{").replace("]", "}").replace("'", '"')
        )
        chain_embeddings_df["feature_timestamp"] = DATE_TO_WRITE
        chain_embeddings_df["country_code"] = self.params.get('country')

        LOG.info("Writing embeddings to bigquery")
        delete_data_for_date(DST_TABLE_NAME, DATE_TO_WRITE, self.params.get('country'))
        write_to_bq(df=chain_embeddings_df, table_name=DST_TABLE_NAME, if_exists="append")
        update_timestamp_column(
            DATE_TO_WRITE.strftime('%Y-%m-%d'),
            DST_TABLE_NAME,
            self.params.get('country')
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Chain embeddings generation for two tower model."
    )
    parser.add_argument(
        "--country",
        required=True,
        choices=['EG', 'AE', 'KW', 'QA', 'OM', 'BH', 'IQ', 'JO'],
        help="Possible country values are: ['EG', 'AE', 'KW', 'QA', 'OM', 'BH', 'IQ', 'JO']"
    )
    args = parser.parse_args()
    chain_embeddings_obj = ChainEmbeddings(country=args.country)
    chain_embeddings_obj()
