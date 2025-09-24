import sys
from pathlib import Path
root_dir = Path(__file__).parent
[sys.path.insert(0, str(root_dir := root_dir.parent)) for _ in range(3)]
import os
import mlflow
import structlog
from base.v1 import mlutils
from dotenv import load_dotenv
import numpy as np


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
        model_dict = mlutils.load_registered_model(
            model_name=self.MODEL_NAME,
            alias='best_model',  # make sure it's matching the alias you set on mlflow server
        )
        mlflow_model = model_dict['mlflow_model']
        model_tags = model_dict['model_tags']
        model_version = model_dict['model_version']

        model_input = {
            'input': np.array([[5.1, 3.5, 1.4, 0.2]])  # --> Set the suitable model input (list, array, etc.)
        }

        predictions = mlflow_model.predict(model_input)

        LOG.info(f"Predictions: {predictions}")


if __name__ == '__main__':
    run_predictions = ModelPredictor()
    run_predictions()
