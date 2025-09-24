from base.v0.mlclass import MlflowBase
import mlflow
import joblib
from typing import Dict
import numpy as np
from projects.sku_forecasting.common.utils import get_training_data, metrics_evaluation
from sklearn.preprocessing import LabelEncoder
import pandas as pd 
import tensorflow as tf 

class ModelWrapper(MlflowBase):
    """
    A wrapper class for the Iris dataset model, inheriting from MlflowBase.
    This class handles model training, prediction, and context loading.
    """
    def __init__(self) -> None:
        """
        Initialize the IrisModelWrapper class.
        """
        self.model_name = 'sku_forecasting_cnn'  # Set the model's name (this what will be logged to mlflow server)
        super().__init__()

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

        preds = self.model.predict(model_input)
         
        return preds
    
    

    @staticmethod
    def get_input(self) -> Dict:
        """
        :return: A sample input for the model.
        """
 

        model_input = {
            'input': None # --> Set the suitable model input (list, array, etc.)
        }
        return model_input
