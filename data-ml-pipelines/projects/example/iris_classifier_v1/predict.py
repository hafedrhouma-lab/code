import structlog
from dotenv import load_dotenv

from base.v0 import mlutils

load_dotenv()

LOG: structlog.stdlib.BoundLogger = structlog.get_logger()


class IrisPredictor:

    def __init__(self):
        pass

    # Add other methods as needed, preprocessing, postprocessing, etc...

    def __call__(self, start_date=None):

        LOG.info("Retrieving the model from mlflow server...")
        self.mlflow_model = mlutils.load_registered_model(
            model_name='example_iris_classifier_v1',
            alias='best_model',
        )

        example_input = self.mlflow_model.unwrap_python_model().get_sample_input()

        LOG.info("Predicting using the model...")
        predictions = self.mlflow_model.predict(example_input)
        LOG.info(f"Predictions: {predictions}")

        # Add your custom logic here to write the predictions to a database or a file, etc...
        return


if __name__ == '__main__':
    predictor = IrisPredictor()
    predictor()

