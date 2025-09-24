import sys
from pathlib import Path
root_dir = Path(__file__).parent
[sys.path.insert(0, str(root_dir := root_dir.parent)) for _ in range(3)]
from base.v0.mlclass import MlflowBase
import mlflow
import joblib
from typing import Dict
import numpy as np


class ModelWrapper(MlflowBase):
    """
    A wrapper class for the Iris dataset model, inheriting from MlflowBase.
    This class handles model training, prediction, and context loading.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Loads the model context for predictions.

        :param context: The context for the Python model.
        """
        self.model = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input):
        """
        Predicts using the loaded model.

        :param context: The context for the Python model.
        :param model_input: A dictionary containing the model input(s).
        :return: The prediction result.
        """
        pred = self.model.predict(model_input['input'])
        return pred

    @staticmethod
    def get_sample_input() -> Dict:
        """
        :return: A sample input for the model.
        """
        model_input = {
            'input': np.array([[5.1, 3.5, 1.4, 0.2]])  # --> Set the suitable model input (list, array, etc.)
        }
        return model_input
