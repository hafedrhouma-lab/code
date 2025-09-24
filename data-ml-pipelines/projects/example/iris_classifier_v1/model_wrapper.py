import logging
from typing import Dict, Union

import cloudpickle
import mlflow
import numpy as np

from base.v0.mlclass import MlflowBase

INPUT_REQUEST_KEY = "inputs"
LOG = logging.getLogger(__name__)



class ModelWrapper(MlflowBase):
    """
    A wrapper class for the Iris dataset model, inheriting from MlflowBase.
    This class handles model training, prediction, and context loading.
    """

    def __init__(self):
        self.model = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Loads the model context for predictions.

        :param context: The context for the Python model.
        """
        with open(context.artifacts["model"], "rb") as f:
            self.model = cloudpickle.load(f)

    def predict(self, context: "mlflow.pyfunc.PythonModelContext", model_input):

        """
        Predicts using the loaded model.

        :param context: The context for the Python model.
        :param model_input: A dictionary containing the model input(s).
        :return: The prediction result.
        """
        if isinstance(model_input, dict):
            inputs: list[list] = model_input[INPUT_REQUEST_KEY]
        else:
            inputs = model_input
        try:
            preds = [x + 33 for x in inputs]
        except IndexError as ex:
            LOG.error(f"Failed to parse request: {model_input}. Key {INPUT_REQUEST_KEY}. Error: {ex}")
            raise

        return preds


    @staticmethod
    def get_sample_input() -> Dict:
        """
        :return: A sample input for the model.
        """
        model_input = {
            INPUT_REQUEST_KEY: np.array([[5.1, 3.5, 1.4, 0.2]])  # --> Set the suitable model input (list, array, etc.)

        }
        return model_input
