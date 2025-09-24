import os
import mlflow
from pathlib import Path
from base.v1 import mlutils
import numpy as np
import pandas as pd
import argparse
import warnings
import geohash
import structlog
from dotenv import load_dotenv
import requests
from datetime import datetime, timedelta, timezone

from projects.vendor_ranking.common.two_towers.src.cli import USER_MODEL_INPUT
from projects.vendor_ranking.common.two_towers.src.data.datasets.query_loader import QueryLoader
from projects.vendor_ranking.common.two_towers.src.utils.eval_utils import (
    get_embeddings_similarity,
    df_to_np_embeddings
)
from projects.vendor_ranking.common.two_towers.src.data import (
    get_data_fetcher,
    get_feature_store
)

from projects.vendor_ranking.two_towers_v1.tests.inputs.ranking import (
    INPUT_CHAINS_COUNTRY,
    ACCOUNT_ID_MAP,
    LATITUDE,
    LONGITUDE
)


queries_path = Path(__file__).parent.resolve() / "queries/"
query_loader = QueryLoader(template_dir=str(queries_path))
ace_api_server_request_path = Path(__file__).parent.resolve() / "ace_api_server_request.sh"

warnings.simplefilter(action='ignore', category=FutureWarning)
load_dotenv(override=True)
LOG: structlog.stdlib.BoundLogger = structlog.get_logger()


def convert_to_lists(input_dict):
    return {
        key: [value] if not isinstance(value, list)
        else value for key, value in input_dict.items()
    }


def convert_embeddings_to_list(embedding_str):
    try:
        return list(map(float, embedding_str.strip('{}').split(', ')))
    except ValueError:
        return None


def get_sorted_chains(user_embeddings, chain_embeddings):
    chains_embeddings_np = df_to_np_embeddings(chain_embeddings)
    similarity = get_embeddings_similarity(user_embeddings, chains_embeddings_np)
    sorted_chains_idx = np.argsort(-similarity)
    sorted_chains = chain_embeddings.index.values[sorted_chains_idx]
    return sorted_chains


rename_dict = {
    "most_recent_10_orders": "user_prev_chains",
    "frequent_chains": "freq_chains",
    "most_recent_15_search_keywords": "prev_searches",
    "most_recent_10_clicks_wo_orders": "prev_clicks",
    "frequent_clicks": "freq_clicks"
}
TEST_DATE = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
GEOHASH_CODE = geohash.encode(LATITUDE, LONGITUDE, precision=6)


class TestPipeline:
    def __init__(self, country, test_date: str, account_id):
        exp_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        self.test_date = test_date
        self.country = country
        self.account_id = account_id

        self.MODEL_NAME = f"{exp_name}_{self.country.lower()}"
        LOG.info(f"Tracking uri: {mlflow.get_tracking_uri()}")
        LOG.info(f"Experiment id: {exp_name}")

        self.model_input = convert_to_lists(USER_MODEL_INPUT)

    def __call__(self):
        LOG.info("Retrieving the model from mlflow server...")
        mlflow_model = mlutils.load_registered_model(
            model_name=self.MODEL_NAME,
            alias='ace_champion_model_staging'
        )
        self.mlflow_model = mlflow_model['mlflow_model']

        # LOCAL EVALUATION
        account_features_query = query_loader.load_query(
            'account_features.sql.j2',
            start_date=self.test_date,
            end_date=self.test_date,
            country_code=self.country,
            account_orders=get_feature_store().get_data_source("account_orders").get_table_query_string(),
            account_id=self.account_id
        )
        df_account_features = get_data_fetcher().fetch_data(
            description='Order Features For train Data',
            source="feast",
            entity_sql=account_features_query,
            features=get_feature_store().get_feature_service('inference_2t_v1_eg_fs')
        )
        df_account_features.rename(columns=rename_dict, inplace=True)
        model_input = df_account_features.squeeze().to_dict()

        model_input.update({
            "order_hour": datetime.now(timezone.utc).hour,
            "order_weekday": datetime.now(timezone.utc).weekday(),
            "delivery_area_id": '2377',
            'geohash6': GEOHASH_CODE
        })

        model_input = convert_to_lists(model_input)
        model_input.pop("event_timestamp", None)
        model_input = {key: value for key, value in model_input.items() if key in USER_MODEL_INPUT}

        model_output = self.mlflow_model.predict(model_input)
        model_output_array = model_output['value']
        user_model_embeddings = model_output_array.tolist()
        user_embeddings_np = np.array(user_model_embeddings)

        embeddings_query = query_loader.load_query(
            'chain_embeddings_bq.sql.j2',
            test_date=self.test_date,
            country_code=self.country
        )
        chain_embeddings = get_data_fetcher().fetch_data(
            description='Two tower Trained model chain embeddings',
            source="sql",
            query=embeddings_query,
        )
        chain_embeddings['embeddings'] = chain_embeddings['embeddings'].apply(
            convert_embeddings_to_list
        )
        chain_embeddings = chain_embeddings.set_index('chain_id')
        chain_embeddings['embeddings'] = chain_embeddings['embeddings'].apply(
            lambda x: np.array(x)
        )

        chain_requested = INPUT_CHAINS_COUNTRY[self.country]
        vendor_ids = [100 + i for i in range(len(chain_requested))]
        mapping_vendor_chains = {
            "vendor_id":  vendor_ids,
            "chain_id": chain_requested
        }
        input_df = pd.DataFrame(mapping_vendor_chains)

        requested_chains = chain_embeddings.loc[
            chain_embeddings.index.isin(input_df['chain_id'])
        ]
        ranked_chains = get_sorted_chains(user_embeddings_np, requested_chains)
        ranked_chains = ranked_chains.tolist()
        vendor_ids = [
            input_df.loc[input_df['chain_id'] == chain_id, 'vendor_id'].values[0]
            for chain_id in ranked_chains
        ]

        #ACE
        url = 'https://ace.talabat.com/v2/vendor_ranking/annotated/sort'
        headers = {
            'User-Agent': 'LoadImpact-Shield-a8bf8e43b33f448c891e9434d40ba69b',
            'Content-Type': 'application/json'
        }
        timestamp = datetime.now(timezone.utc).isoformat(timespec='microseconds')
        payload = {
            "timestamp": timestamp,
            "device_source": 0,
            "app_version": "",
            "customer_id": self.account_id,
            "model_nickname": "two_towers_v3",
            "locale": "en-US",
            "location": {
                "country_id": 4,
                "country_code": self.country,
                "city_id": 35,
                "area_id": 2377,
                "latitude": LATITUDE,
                "longitude": LONGITUDE
            },
            "vendors_df": {
                "chain_id": input_df['chain_id'].tolist(),
                "vendor_id": input_df['vendor_id'].tolist(),
                "delivery_fee": [0] * len(input_df['chain_id'].tolist()),
                "vendor_rating": [3] * len(input_df['chain_id'].tolist()),
                "delivery_time": [20] * len(input_df['chain_id'].tolist()),
                "status": ["0"] * len(input_df['chain_id'].tolist()),
                "min_order_amount": [0] * len(input_df['chain_id'].tolist()),
                "has_promotion": [False] * len(input_df['chain_id'].tolist())
            }
        }
        response = requests.post(url, headers=headers, json=payload)
        ace_response_data = response.json()

        LOG.info(f"ACE output: :{ace_response_data}")
        LOG.info(f"Local evaluation's output: :{vendor_ids}")

        assert ace_response_data[:30] == vendor_ids[:30], "Ace output and local evaluation mismatch"

        LOG.info(
            f"First 30 Ace output and model's output evaluated locally are matching"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test two tower model.")
    parser.add_argument(
        "--country",
        required=True,
        choices=['AE', 'KW', 'QA', 'OM', 'BH', 'IQ', 'JO'],
        help="Possible country values are: ['AE', 'KW', 'QA', 'OM', 'BH', 'IQ', 'JO']"
    )
    args = parser.parse_args()
    test_pipeline = TestPipeline(
        args.country,
        TEST_DATE,
        ACCOUNT_ID_MAP[args.country]
    )
    test_pipeline()

