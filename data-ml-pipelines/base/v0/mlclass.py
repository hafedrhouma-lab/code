from abc import ABC, abstractmethod
from typing import Any, Dict
import structlog
import mlflow.pyfunc
from mlflow.pyfunc import PythonModelContext


# Configure structlog for logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

LOG: structlog.stdlib.BoundLogger = structlog.get_logger()


class MlflowBase(mlflow.pyfunc.PythonModel, ABC):
    """
    Base class for MLflow Python models.

    This class provides a structure for loading and predicting using models
    in the MLflow framework. It is designed to be subclassed with specific
    implementations for model loading and prediction.
    """

    @abstractmethod
    def load_context(self, context: PythonModelContext) -> None:
        """
        Abstract method to load the model context for predictions.

        This method should be implemented by subclasses to load any necessary
        information for making predictions with the model.

        :param context: The context for the Python model.
        """
        pass

    @abstractmethod
    def predict(self, context: PythonModelContext, model_input: Dict[str, Any]) -> Any:
        """
        Abstract method to make predictions using the loaded model.

        This method should be implemented by subclasses to perform predictions
        based on the model and input data.

        :param context: The context for the Python model.
        :param model_input: A dictionary containing the input data for the model.
        :return: The prediction result.
        """
        pass

    def test_model(self, run):
        """
        Test the model using the provided run_uri.
        :param run: The mlflow run object
        :return: None
        """
        LOG.info("---Model Testing---")
        run_uri = f'runs:/{run.info.run_id}/model'
        loaded_model = mlflow.pyfunc.load_model(run_uri)
        model_input = self.get_sample_input()
        prediction = loaded_model.predict(model_input)
        LOG.info(f"Prediction: {prediction}")
        LOG.info("Model tested successfully!")
