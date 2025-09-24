import os
import mlflow
import structlog
from base.v0 import mlutils
from dotenv import load_dotenv
import numpy as np

load_dotenv()

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
            model_name='example_iris_classifier_v0',   # make sure it's matching the model name you set on mlflow server
            alias='best_model',  # make sure it's matching the alias you set on mlflow server
        )

        model_input = {
            'input': np.array([[5.1, 3.5, 1.4, 0.2]])  # --> Set the suitable model input (list, array, etc.)
        }

        predictions = self.mlflow_model.predict(model_input)

        LOG.info(f"Predictions: {predictions}")

if __name__ == '__main__':
    run_predictions = ModelPredictor()
    run_predictions()
