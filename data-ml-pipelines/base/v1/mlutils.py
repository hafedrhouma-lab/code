import structlog
from mlflow.tracking import MlflowClient
import mlflow
import tempfile
import shutil
import sys
import os
from contextlib import contextmanager

LOG: structlog.stdlib.BoundLogger = structlog.get_logger()


@contextmanager
def temporarily_modify_sys_path(paths_to_add):
    original_sys_path = sys.path.copy()
    try:
        # Add specified paths
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)

        yield
    finally:
        # Restore original sys.path
        sys.path = original_sys_path
        LOG.info("sys.path restored to original state.")


def load_registered_model(model_name, alias, use_remote_code=True):
    """
    Load a registered model version with optional control over code dependencies.

    :param model_name: The name of the registered model.
    :param alias: The alias of the model version.
    :param use_remote_code: If True (default), use the code packaged with the model;
                            if False, use local code.
    :return: A dictionary containing the model object and metadata.
    """
    client = MlflowClient()

    # Get model version information
    model_details = client.get_model_version_by_alias(model_name, alias)
    model_version = model_details.version
    run_id = model_details.run_id
    run_info = client.get_run(run_id)
    experiment_id = run_info.info.experiment_id
    experiment = client.get_experiment(experiment_id)
    LOG.info(f"Model Details:")
    LOG.info(f"Version: {model_version}")
    LOG.info(f"Tags: {model_details.tags}")
    LOG.info(f"Experiment Details:")
    LOG.info(f"Experiment Name: {experiment.name}")
    LOG.info(f"Experiment ID: {experiment_id}")
    LOG.info(f"Run Name: {run_info.info.run_name}")
    LOG.info(f"Run ID: {run_id}")
    LOG.info(f"Artifact Location: {experiment.artifact_location}")

    # Build model URI
    model_uri = f"models:/{model_name}@{alias}"

    if use_remote_code:
        temp_dir = tempfile.mkdtemp()
        try:
            # Download the model artifacts
            local_path = mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri, dst_path=temp_dir
            )
            LOG.debug(f"Model artifacts downloaded to: {local_path}")

            code_dir = os.path.join(local_path, 'code')
            if os.path.exists(code_dir):
                # Prepare paths to add
                paths_to_add = []

                # Add code_dir and local_path to paths_to_add
                paths_to_add.extend([code_dir])
                LOG.debug(f"Temporarily adding remote code to sys.path: {code_dir}")

                # Search for the directory containing 'model_wrapper.py'
                model_wrapper_dir = None
                for root, dirs, files in os.walk(code_dir):
                    if 'model_wrapper.py' in files:
                        model_wrapper_dir = root
                        break

                if model_wrapper_dir:
                    paths_to_add.insert(0, model_wrapper_dir)
                    LOG.debug(f"Temporarily adding remote code to sys.path: {model_wrapper_dir}")

                    # Ensure __init__.py files exist in all directories
                    relative_path = os.path.relpath(model_wrapper_dir, local_path)
                    path_parts = relative_path.split(os.sep)
                    current_dir = local_path
                    for part in path_parts:
                        current_dir = os.path.join(current_dir, part)
                        init_file = os.path.join(current_dir, '__init__.py')
                        if not os.path.exists(init_file):
                            open(init_file, 'a').close()

                else:
                    LOG.warning(f"'model_wrapper.py' not found under code directory.")

                # Use the context manager to modify sys.path temporarily
                with temporarily_modify_sys_path(paths_to_add=paths_to_add):
                    # Remove 'model_wrapper' from sys.modules
                    if 'model_wrapper' in sys.modules:
                        del sys.modules['model_wrapper']
                        LOG.debug("Removed 'model_wrapper' from sys.modules")

                    # Load the model from local_path
                    model = mlflow.pyfunc.load_model(model_uri=local_path)
            else:
                LOG.warning(f"No 'code' directory found in model artifacts at {code_dir}")
                model = mlflow.pyfunc.load_model(model_uri=local_path)
                LOG.info(f"Model loaded from remote code")
        finally:
            shutil.rmtree(temp_dir)
    else:
        # Use local code
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        LOG.info(f"Model loaded from local code")

    return {
        "mlflow_model": model,
        "model_version": model_version,
        "model_tags": model_details.tags,
        "run_info": run_info,
        "experiment": experiment,
    }


def load_experiment_model(run_id):
    # Assuming the model is logged under the artifact path "model"
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)

    return model


def get_latest_run(experiment_name):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = client.search_runs(experiment.experiment_id)
    latest_run = runs[0]
    return latest_run


def register_model(model_name, run, alias, tags=None):
    """Register a model version with an alias and tags.
    :param model_name: The name of the registered model.
    :param run: The run object.
    :param alias: The alias of the model version.
    :param tags: A dictionary of tags to add to the model version.
    """

    LOG.info("---Model Registering---")
    # if alias not in allowed_aliases:
    #     raise ValueError(f"Alias '{alias}' not found in aliases: {allowed_aliases}")

    client = MlflowClient()
    model_uri = f"runs:/{run.info.run_id}/model"
    try:
        registered_model = client.get_registered_model(model_name)
    except Exception:
        LOG.info(f"Registering a new model '{model_name}'")
        client.create_registered_model(model_name)
        registered_model = client.get_registered_model(model_name)

    LOG.info(f"Name: {registered_model.name}")
    # Create a new model version within the specified registered model
    model_version = client.create_model_version(model_name, model_uri, run.info.run_id)
    LOG.info(f"Version: {model_version.version}")
    client.set_registered_model_alias(name=model_name, alias=alias, version=model_version.version)

    # Get a model version by alias
    alias_model_version = client.get_model_version_by_alias(name=model_name, alias=alias)
    LOG.info(f"Aliases: {alias_model_version.aliases}")

    # Check if tags are a dict, then Add tags to the model version
    if tags:
        if isinstance(tags, dict):
            for key, value in tags.items():
                client.set_model_version_tag(model_name, model_version.version, key, value)
        else:
            raise ValueError("Tags must be a dictionary")


def get_latest_model_version_run_tags(model_name):
    client = MlflowClient()

    # Get the latest version of the model
    latest_version_info = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version_info:
        raise ValueError(f"No versions found for model '{model_name}'")

    latest_version = latest_version_info[0]

    # Extract run ID from the latest model version
    run_id = latest_version.run_id

    # Fetch the run data
    run = client.get_run(run_id)
    # feature_service_name = run.data.tags.get("feature_service")
    LOG.info(f"Model Version: {latest_version.version}")

    return latest_version.version


def load_model_by_alias(model_name, alias):
    # Construct the model URI using the model name and alias
    model_uri = f"models:/{model_name}@{alias}"

    # Load the model using MLflow
    model = mlflow.pyfunc.load_model(model_uri)

    return model
