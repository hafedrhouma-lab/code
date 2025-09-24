import sys
from pathlib import Path
root_dir = Path(__file__).parent
[sys.path.insert(0, str(root_dir := root_dir.parent)) for _ in range(3)]
import os
import numpy as np
from base.v0.db_utils import BigQuery
# Temporarily replace np.object with object to avoid deprecation errors
try:
    np.object = object
except AttributeError:
    pass
import mlflow
import structlog
from base.v1 import mlutils
from dotenv import load_dotenv
import joblib


load_dotenv(override=True)
LOG: structlog.stdlib.BoundLogger = structlog.get_logger()


class ModelPredictor:
    def __init__(self):
        exp_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        self.MODEL_NAME = exp_name
        LOG.info(f"Tracking uri: {mlflow.get_tracking_uri()}")
        LOG.info(f"Model name: {exp_name}")

    def get_inference_data(self, batch_size=1000):
        # # replace with actual inference data
        # bq = BigQuery()
        # query = """
        # SELECT * from `` LIMIT 5
        # """
        # df = bq.read(query)

        inference_tensor = self.mlflow_model.unwrap_python_model().get_sample_input()


        return inference_tensor

    @staticmethod
    def push_to_bq(df_output):
        bq = BigQuery()
        bq.write(df_output, table_name='tlb-data-prod.adex_gem_minimize_popup.predictions', if_exist_action='append')

    def __call__(self):

        LOG.info("Retrieving the model from mlflow server...")
        self.model_dict = mlutils.load_registered_model(
            model_name=self.MODEL_NAME,   # make sure it's matching the model name you set on mlflow server
            alias='best_model',  # make sure it's matching the alias you set on mlflow server
        )
        #
        self.mlflow_model = self.model_dict['mlflow_model']
        self.model_tags = self.model_dict['model_tags']
        self.model_version = self.model_dict['model_version']


        # # For loop starts here --> for every batch of data
        model_input = self.get_inference_data()

        predictions = self.mlflow_model.predict(model_input)
        LOG.info(f"Predictions: {predictions}")
        # self.push_to_bq(predictions)
        # # For loop ends here
        return self.model_dict

if __name__ == '__main__':
    run_predictions = ModelPredictor()
    run_predictions()
