import sys
from pathlib import Path
root_dir = Path(__file__).parent
[sys.path.insert(0, str(root_dir := root_dir.parent)) for _ in range(3)]
from base.v0.mlclass import MlflowBase
import mlflow
from typing import Dict
import numpy as np
import tensorflow as tf
import struct2tensor.ops.gen_decode_proto_sparse


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
        self.model = tf.saved_model.load(context.artifacts['model'])
        self.infer = self.model.signatures['serving_default']

    def predict(self, context, model_input):
        """
        Predicts using the loaded model.

        :param context: The context for the Python model.
        :param model_input: A dictionary containing the model input(s).
        :return: The prediction result.
        """
        # Run inference using the serialized example

        input_tensor = model_input['input']
        output = self.infer(inputs=input_tensor)
        return output

    def get_sample_input(self) -> Dict[str, tf.Tensor]:
        """
        :return: A sample input for the model.
        """
        serialized_example = self.create_example()
        input_tensor = tf.constant([serialized_example])
        model_input = {
            'input': input_tensor
        }
        return model_input

    @staticmethod
    def create_example():
        # Create a dictionary with random data for the features
        # Replace these with actual feature names and appropriate data if available
        feature = {
            "feature_1": tf.train.Feature(float_list=tf.train.FloatList(value=[np.random.rand()])),
            "feature_2": tf.train.Feature(int64_list=tf.train.Int64List(value=[np.random.randint(0, 100)])),
            "feature_3": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'random_string']))
            # Add other features as needed
        }

        # Create the Example from the features
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize the Example to string
        serialized_example = example_proto.SerializeToString()

        return serialized_example

