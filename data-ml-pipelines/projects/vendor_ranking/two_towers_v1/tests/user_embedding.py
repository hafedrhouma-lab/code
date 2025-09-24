import os
import numpy as np
import json
import requests
from base.v1 import mlutils
import warnings
import mlflow
import structlog
from dotenv import load_dotenv
import argparse

from projects.vendor_ranking.common.two_towers.src.cli import USER_MODEL_INPUT

warnings.simplefilter(action='ignore', category=FutureWarning)
load_dotenv(override=True)
LOG: structlog.stdlib.BoundLogger = structlog.get_logger()


def convert_to_lists(input_dict):
    return {
        key: [value] if not isinstance(value, list)
        else value for key, value in input_dict.items()
    }


class TestPipeline:
    def __init__(self, country):
        exp_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        self.country = country
        self.MODEL_NAME = f"{exp_name}_{self.country.lower()}"
        LOG.info(f"Tracking uri: {mlflow.get_tracking_uri()}")
        LOG.info(f"Experiment id: {exp_name}")

        self.model_input = convert_to_lists(USER_MODEL_INPUT)
        self.ace_request_input = {
            "inputs": convert_to_lists(USER_MODEL_INPUT)
        }
        self.ace_url = f"https://ace.talabat.com/models/vendor-ranking-tt-v1/{self.country.lower()}/invocations"
        self.headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
            'User-Agent': 'curl/7.64.1'
        }
        self.tolerance = 1e-5

    def __call__(self):
        LOG.info("Retrieving the model from mlflow server...")
        mlflow_model = mlutils.load_registered_model(
            model_name=self.MODEL_NAME,
            alias='ace_champion_model_staging'
        )
        self.mlflow_model = mlflow_model['mlflow_model']
        model_output = self.mlflow_model.predict(self.model_input)
        model_output_array = model_output['value']
        model_embeddings = model_output_array.tolist()

        ace_response = requests.post(
            self.ace_url,
            headers=self.headers,
            data=json.dumps(self.ace_request_input)
        )
        ace_response_data = ace_response.json()
        ace_embeddings = ace_response_data['predictions']['value']

        model_embeddings_array = np.array(model_embeddings)
        ace_embeddings_array = np.array(ace_embeddings)
        are_similar = np.all(
            np.abs(model_embeddings_array - ace_embeddings_array) < self.tolerance
        )

        assert are_similar
        LOG.info(
            f"Are the embeddings similar within a tolerance of {self.tolerance}? {are_similar}"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train two tower model.")
    parser.add_argument(
        "--country",
        required=True,
        choices=['AE', 'KW', 'QA', 'OM', 'BH', 'IQ', 'JO'],
        help="Possible country values are: ['AE', 'KW', 'QA', 'OM', 'BH', 'IQ', 'JO']"
    )
    args = parser.parse_args()
    test_pipeline = TestPipeline(country=args.country)
    test_pipeline()