import sys
from pathlib import Path
root_dir = Path(__file__).parent
[sys.path.insert(0, str(root_dir := root_dir.parent)) for _ in range(3)]
import os
import mlflow
import structlog
from base.v0 import mlutils
from dotenv import load_dotenv
load_dotenv(override=True)

LOG: structlog.stdlib.BoundLogger = structlog.get_logger()


class ModelPredictor:
    def __init__(self):
        exp_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        self.MODEL_NAME = exp_name
        LOG.info(f"Tracking uri: {mlflow.get_tracking_uri()}")
        LOG.info(f"Experiment id: {exp_name}")

    def get_inference_data(self):
        pass

    def preprocess_inference_data(self, data):
        pass

    @staticmethod
    def push_to_bq(df_output):
        pass

    def __call__(self):

        LOG.info("Retrieving the model from mlflow server...")
        self.mlflow_model = mlutils.load_registered_model(
            model_name='vendor_ranking_session_based_ranker',   # make sure it's matching the model name you set on mlflow server
            alias='best_model',  # make sure it's matching the alias you set on mlflow server
        )

        inference_data = self.get_inference_data()
        inference_data = self.preprocess_inference_data(inference_data)

        model_input = {
            'model_input': inference_data,
        }

        predictions = self.mlflow_model.predict(model_input)

        self.push_to_bq(predictions)


if __name__ == '__main__':
    run_predictions = ModelPredictor()
    run_predictions()
