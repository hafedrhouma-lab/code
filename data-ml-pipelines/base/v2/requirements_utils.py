import mlflow
import sys
import yaml
from mlflow.models import infer_pip_requirements
from pathlib import Path
import structlog
from typing import Optional, Union
from mlflow.utils import (
    PYTHON_VERSION,
    get_major_minor_py_version
)
from mlflow.utils.requirements_utils import (
    _parse_requirements,
    _check_requirement_satisfied
)

_logger: structlog.stdlib.BoundLogger = structlog.get_logger()

def load_python_env(env_path: Union[str, Path]):
    """
    Load and process a python_env.yaml file to extract Python version.

    Args:
        pip_requirements (Union[str, Path]): Path to the requirements.txt file.

    Returns:
        Tuple[str, List[str]]: Python version and a list of dependencies with versions.
    """
    if not Path(env_path).exists():
        raise FileNotFoundError(f"Python environment file not found at {env_path}")

    with open(env_path, "r") as file:
        python_env = yaml.safe_load(file)

    python_version = python_env.get("python", "Unknown")
    build_dependencies = python_env.get("build_dependencies", [])

    return python_version



def error_dependency_requirement_mismatches(model_requirements):
    try:
        _logger.info("Checking for dependency mismatches...")
        _logger.info(f"Model requirements: {model_requirements}")
        mismatch_infos = []
        for req in model_requirements:
            mismatch_info = _check_requirement_satisfied(req)
            if mismatch_info is not None:
                mismatch_infos.append(str(mismatch_info))

        if len(mismatch_infos) > 0:
            mismatch_str = " - " + "\n - ".join(mismatch_infos)
            warning_msg = (
                "Detected one or more mismatches between the model's dependencies and the current "
                f"Python environment:\n{mismatch_str}\n"
                "To fix the mismatches, call `mlflow.pyfunc.get_model_dependencies(model_uri)` "
                "to fetch the model's environment and install dependencies using the resulting "
                "environment file."
            )
            _logger.error(warning_msg)
            raise EnvironmentError(warning_msg)

    except Exception as e:
        _logger.error(
            f"Encountered an unexpected error ({e!r}) while detecting model dependency "
            "mismatches. Set logging level to DEBUG to see the full traceback."
        )
        _logger.debug("", exc_info=True)
        raise EnvironmentError(f"requirements mismatches.")

def validate_model_env(pip_requirements):
    _logger.info("Env validation started...")
    # Load and validate specified dependencies
    if pip_requirements:
        if not Path(pip_requirements).exists():
            raise FileNotFoundError(f"Requirements file not found at {pip_requirements}")

        env_path = Path(pip_requirements).parent / "python_env.yaml"
        logged_python_version = load_python_env(env_path)
    else:
        raise ValueError("Path to the requirements.txt file must be specified to check dependencies.")

    # Validate Python version consistency
    local_python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    _logger.info(f"Local Python version: {local_python_version}")
    _logger.info(f"python_env.yaml Python version: {logged_python_version}")

    if logged_python_version and local_python_version not in logged_python_version:
        _logger.error(
            f"Local Python version {local_python_version} does not match the (python_env.yaml) Python version {logged_python_version}.")
        raise EnvironmentError(
            f"Local Python version {local_python_version} does not match the (python_env.yaml) Python version {logged_python_version}.")

    # Check for missing dependencies
    model_requirements = mlflow.pyfunc._get_pip_requirements_from_model_path(Path(pip_requirements).parent)
    error_dependency_requirement_mismatches(model_requirements)



def _error_potentially_incompatible_py_version_if_necessary(model_py_version=None):
    """
    Compares the version of Python that was used to save a given model with the version
    of Python that is currently running. If a major or minor version difference is detected,
    logs an appropriate warning.
    """
    _logger.info("Checking for Python version mismatch...")
    if model_py_version is None:
        _logger.error(
            "The specified model does not have a specified Python version. It may be"
            " incompatible with the version of Python that is currently running: Python %s",
            PYTHON_VERSION,
        )
        raise EnvironmentError("Model does not have a specified Python version.")

    elif get_major_minor_py_version(model_py_version) != get_major_minor_py_version(PYTHON_VERSION):
        _logger.error(
            "The version of Python that the model was saved in, `Python %s`, differs"
            " from the version of Python that is currently running, `Python %s`,"
            " and may be incompatible",
            model_py_version,
            PYTHON_VERSION,
        )
        raise EnvironmentError("Model Python version mismatch.")
