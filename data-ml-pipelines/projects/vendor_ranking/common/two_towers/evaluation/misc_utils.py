import mlflow
import structlog

from base.v1.mlutils import load_registered_model

LOG: "structlog.stdlib.BoundLogger" = structlog.getLogger(__name__)


def load_mlflow_model(model_name, alias):
    """
    Retrieve and unwrap a model from MLflow server based on the experiment name and country.

    Args:
        exp_name (str): The MLflow experiment name.
        country (str): The country code to identify the model.
        alias (str): The alias to use when retrieving the model version. Default is 'ace_champion_model_staging'.

    Returns:
        Python model: The unwrapped Python model.
    """
    LOG.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
    LOG.info(f"Retrieving the model '{model_name}' from MLflow server...")

    model_dict = load_registered_model(
        model_name=model_name,
        alias=alias
    )
    return model_dict['mlflow_model'].unwrap_python_model()
