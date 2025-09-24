from pathlib import Path
import pandas as pd
import argparse
import warnings
import structlog
from dotenv import load_dotenv
import requests
from datetime import datetime, timezone


from projects.vendor_ranking.two_towers_v1.tests.inputs.ranking import (
    INPUT_CHAINS_COUNTRY,
    INPUT_CHAINS_COUNTRY_NOT_EXISITNG,
    ACCOUNT_ID_MAP,
    LATITUDE,
    LONGITUDE
)

ace_api_server_request_path = Path(__file__).parent.resolve() / "ace_api_server_request.sh"

warnings.simplefilter(action='ignore', category=FutureWarning)
load_dotenv(override=True)
LOG: structlog.stdlib.BoundLogger = structlog.get_logger()


class TestPipeline:
    def __init__(self, country, account_id):
        self.country = country
        self.account_id = account_id

    def __call__(self):
        chain_requested_embeddings = INPUT_CHAINS_COUNTRY[self.country]
        chains_requested_no_embeddings = INPUT_CHAINS_COUNTRY_NOT_EXISITNG[self.country]

        chain_requested = chain_requested_embeddings + chains_requested_no_embeddings

        vendor_ids = [100 + i for i in range(len(chain_requested))]
        mapping_vendor_chains = {
            "vendor_id":  vendor_ids,
            "chain_id": chain_requested
        }
        input_df = pd.DataFrame(mapping_vendor_chains)

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

        LOG.info(f"4 vendors not for which chains are not in request: :{vendor_ids[-4:]}")
        LOG.info(f"ACE output: :{ace_response_data}")

        assert ace_response_data[-4:] == vendor_ids[-4:], \
        "Chains with no embeddings in the request are not appended to the end of ACE reponse"

        LOG.info(
            f"Chains with no embeddings in the request are correctly appended to the end of ACE reponse"
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
        ACCOUNT_ID_MAP[args.country]
    )
    test_pipeline()

