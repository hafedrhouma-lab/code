import importlib
import structlog
import os
from pathlib import Path
from typing import Any, Optional, Union
import mlflow
from mlflow.pyfunc import PyFuncModel
from mlflow.pyfunc import _get_pip_requirements_from_model_path
from mlflow.environment_variables import (
    _MLFLOW_IN_CAPTURE_MODULE_PROCESS,
)
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.dependencies_schemas import (
    _clear_dependencies_schemas,
)
from mlflow.models.model import (
    _DATABRICKS_FS_LOADER_MODULE,
    MLMODEL_FILE_NAME,
    MODEL_CONFIG,
)
from mlflow.protos.databricks_pb2 import (
    BAD_REQUEST,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    Entity,
    Job,
    LineageHeaderInfo,
    Notebook,
)
from mlflow.tracing.provider import trace_disabled
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import (
    databricks_utils,
)
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_overridden_pyfunc_model_config,
    _validate_and_get_model_config_from_file,
)

from .requirements_utils import (
    error_dependency_requirement_mismatches,
    _error_potentially_incompatible_py_version_if_necessary,
    validate_model_env,

)

try:
    from pyspark.sql import DataFrame as SparkDataFrame

    HAS_PYSPARK = True
except ImportError:
    HAS_PYSPARK = False

FLAVOR_NAME = "python_function"
MAIN = "loader_module"
CODE = "code"
DATA = "data"
ENV = "env"

PY_VERSION = "python_version"

_logger: structlog.stdlib.BoundLogger = structlog.get_logger()


@trace_disabled  # Suppress traces while loading model
def talabat_load_model(
        model_uri: str,
        suppress_warnings: bool = False,
        dst_path: Optional[str] = None,
        model_config: Optional[Union[str, Path, dict[str, Any]]] = None,
) -> PyFuncModel:
    """
    Load a model stored in Python function format.

    Args:
        model_uri: The location, in URI format, of the MLflow model. For example:

            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
            - ``models:/<model_name>/<model_version>``
            - ``models:/<model_name>/<stage>``
            - ``mlflow-artifacts:/path/to/model``

            For more information about supported URI schemes, see
            `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
            artifact-locations>`_.
        suppress_warnings: If ``True``, non-fatal warning messages associated with the model
            loading process will be suppressed. If ``False``, these warning messages will be
            emitted.
        dst_path: The local filesystem path to which to download the model artifact.
            This directory must already exist. If unspecified, a local output
            path will be created.
        model_config: The model configuration to apply to the model. The configuration will
            be available as the ``model_config`` property of the ``context`` parameter
            in :func:`PythonModel.load_context() <mlflow.pyfunc.PythonModel.load_context>`
            and :func:`PythonModel.predict() <mlflow.pyfunc.PythonModel.predict>`.
            The configuration can be passed as a file path, or a dict with string keys.

            .. Note:: Experimental: This parameter may change or be removed in a future
                release without warning.
    """

    lineage_header_info = None
    if (
            not _MLFLOW_IN_CAPTURE_MODULE_PROCESS.get()
    ) and databricks_utils.is_in_databricks_runtime():
        entity_list = []
        # Get notebook id and job id, pack them into lineage_header_info
        if databricks_utils.is_in_databricks_notebook() and (
                notebook_id := databricks_utils.get_notebook_id()
        ):
            notebook_entity = Notebook(id=notebook_id)
            entity_list.append(Entity(notebook=notebook_entity))

        if databricks_utils.is_in_databricks_job() and (job_id := databricks_utils.get_job_id()):
            job_entity = Job(id=job_id)
            entity_list.append(Entity(job=job_entity))

        lineage_header_info = LineageHeaderInfo(entities=entity_list) if entity_list else None

    local_path = _download_artifact_from_uri(
        artifact_uri=model_uri, output_path=dst_path, lineage_header_info=lineage_header_info
    )

    model_requirements = _get_pip_requirements_from_model_path(local_path)
    # Check for potential requirement mismatches, and raise an exception if mismatches are found
    error_dependency_requirement_mismatches(model_requirements)

    model_meta = Model.load(os.path.join(local_path, MLMODEL_FILE_NAME))

    conf = model_meta.flavors.get(FLAVOR_NAME)
    if conf is None:
        raise MlflowException(
            f'Model does not have the "{FLAVOR_NAME}" flavor',
            RESOURCE_DOES_NOT_EXIST,
        )
    model_py_version = conf.get(PY_VERSION)

    # Check for potential Python version mismatch, and raise an exception if a mismatch is found
    _error_potentially_incompatible_py_version_if_necessary(model_py_version=model_py_version)

    _add_code_from_conf_to_system_path(local_path, conf, code_key=CODE)
    data_path = os.path.join(local_path, conf[DATA]) if (DATA in conf) else local_path

    if isinstance(model_config, str):
        model_config = _validate_and_get_model_config_from_file(model_config)

    model_config = _get_overridden_pyfunc_model_config(
        conf.get(MODEL_CONFIG, None), model_config, _logger
    )

    try:
        if model_config:
            model_impl = importlib.import_module(conf[MAIN])._load_pyfunc(data_path, model_config)
        else:
            model_impl = importlib.import_module(conf[MAIN])._load_pyfunc(data_path)
    except ModuleNotFoundError as e:
        # This error message is particularly for the case when the error is caused by module
        # "databricks.feature_store.mlflow_model". But depending on the environment, the offending
        # module might be "databricks", "databricks.feature_store" or full package. So we will
        # raise the error with the following note if "databricks" presents in the error. All non-
        # databricks moduel errors will just be re-raised.
        if conf[MAIN] == _DATABRICKS_FS_LOADER_MODULE and e.name.startswith("databricks"):
            raise MlflowException(
                f"{e.msg}; "
                "Note: mlflow.pyfunc.load_model is not supported for Feature Store models. "
                "spark_udf() and predict() will not work as expected. Use "
                "score_batch for offline predictions.",
                BAD_REQUEST,
            ) from None
        raise e
    finally:
        # clean up the dependencies schema which is set to global state after loading the model.
        # This avoids the schema being used by other models loaded in the same process.
        _clear_dependencies_schemas()
    predict_fn = conf.get("predict_fn", "predict")
    streamable = conf.get("streamable", False)
    predict_stream_fn = conf.get("predict_stream_fn", "predict_stream") if streamable else None

    return PyFuncModel(
        model_meta=model_meta,
        model_impl=model_impl,
        predict_fn=predict_fn,
        predict_stream_fn=predict_stream_fn,
    )


def talabat_log_model(artifact_path, python_model, pip_requirements, input_example=None, **kwargs):
    # Validate the model environment before logging
    validate_model_env(pip_requirements)
    _logger.info("Model environment checks passed.")
    # Log the model if all checks pass
    return  mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=python_model,
        pip_requirements=pip_requirements,
        input_example=input_example,
        **kwargs
    )
