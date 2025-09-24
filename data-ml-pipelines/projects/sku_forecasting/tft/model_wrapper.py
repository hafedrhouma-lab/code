import structlog

from base.v0.mlclass import MlflowBase
import mlflow
import cloudpickle
from typing import Dict
import pandas as pd


LOG: structlog.stdlib.BoundLogger = structlog.get_logger()


class ModelWrapper(MlflowBase):
    """
    A wrapper class for the Iris dataset model, inheriting from MlflowBase.
    This class handles model training, prediction, and context loading.
    """
    def __init__(self) -> None:
        """
        Initialize the IrisModelWrapper class.
        """
        self.model_name = 'sku_forecasting_tft'
        super().__init__()

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Loads the model context for predictions.

        :param context: The context for the Python model.
        """
        LOG.info("-- Loading model context --")
        LOG.info("loading TFT model")
        LOG.info(context.artifacts["model"])
        with open(context.artifacts["model"], "rb") as f:
            self.model = cloudpickle.load(f)
        LOG.info("Succesfully loaded the model")

        LOG.info("loading TFT parameter")
        with open(context.artifacts["tft_parameters"], "rb") as f:
            self.tft_parameters = cloudpickle.load(f)
        LOG.info("Succesfully loaded tft_parameters")

        LOG.info("loading unique categories")
        self.unique_categories = pd.read_csv(context.artifacts["unique_categories"])
        LOG.info("Succesfully loaded the unique categories")

    def predict(self, context, model_input):
        """
        Predicts using the loaded model.

        :param context: The context for the Python model.
        :param model_input: A dictionary containing the model input(s).
        :return: The prediction result.
        """

        inference_data = model_input['inference_data']

        LOG.info("Predicting...")
        predictions = self.model.predict(inference_data, mode="raw", return_x=True)

        return predictions


    @staticmethod
    def get_sample_input() -> Dict:
        """
        :return: A sample input for the model.
        """
        model_input = {
            'inference_data': None

        }
        return model_input
