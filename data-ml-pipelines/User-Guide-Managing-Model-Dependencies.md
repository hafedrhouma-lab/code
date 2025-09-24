# User Guide: Customized MLflow Functions for Managing Model Dependencies

Table of Contents

- [Overview](#overview)
- [talabat_log_model](#1-talabat_log_model)
  - [Purpose](#purpose)
  - [Key Features](#key-features)
  - [Parameters](#parameters)
  - [Requirements](#requirements)
  - [Example Usage](#example-usage)
  - [Behavior](#behavior)
- [load_registered_model and load_experiment_model](#2-load_registered_model-and-load_experiment_model)
  - [Purpose](#purpose-1)
  - [Key Features](#key-features-1)
  - [Usage](#usage)
  - [Behavior](#behavior-1)
- [Expected Benefits](#expected-benefits)

## Overview

To enhance dependency management and ensure consistency between local environments and logged model environments, we introduce two new customized functions for MLflow: `talabat_log_model` and updated versions of `load_model`. These functions enforce strict environment checks to improve the reliability and reproducibility of MLflow models.

### 1. talabat_log_model

#### Purpose

The `talabat_log_model` function is a customized version of [`mlflow.pyfunc.log_model`](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.log_model). It ensures that the local active environment matches the environment defined in the project files before logging a model. If a mismatch is detected, the function will prevent the model from being logged.

#### Key Features

- Verifies that the local environment matches the environment defined in the project’s `python_env.yaml` and `requirements.txt`.
- Prevents models from being logged if the environment is mismatched.
- Encourages the use of `input_example` for better dependency inference during logging.
- Passes all parameters to `mlflow.pyfunc.log_model` for normal behavior when the environment matches.

#### Parameters


Parameter | 	Description 
--- | ---
pip_requirements| 		Full path to the requirements.txt file. **This file must be in the same directory as the python_env.yaml file.**
Other Parameters| 		All other arguments supported by mlflow.pyfunc.log_model, such as artifact_path, python_model, artifacts, code_paths, and input_example.


#### Requirements

##### 1- pip_requirements:
- The full path to the `requirements.txt` file must be provided to the `talabat_log_model` function.


##### 2-  python_env.yaml File:
  - Must be present in the same directory as requirements.txt.
  - Defines the Python version, build dependencies, and package dependencies.
  - Example format:
    ``` yaml
    python: "3.11"
    build_dependencies:
      - pip
      - wheel
    dependencies:
      - -r requirements.txt
    ```


##### 3- input_example:

 - Recommended but optional.
 - Adding an `input_example` improves MLflow’s dependency inference by performing a sample model prediction.

#### Example Usage

``` python
from base.v2.talabat_pyfunc import talabat_log_model

# Log a model using talabat_log_model
model_info = talabat_log_model(
    artifact_path="model",
    python_model=ModelWrapper(),
    artifacts={
        "model": model_pkl_file,
        # Add more artifacts as needed
    },
    code_paths=[
        tmp_dir / "projects",
        tmp_dir / "base"
    ],
    pip_requirements=f"{FILE_PATH}/requirements.txt",
    input_example=[5.1, 3.5, 1.4, 0.2]
)
```

#### Behavior

**If the environment matches:**
  - The function passes all arguments to mlflow.pyfunc.log_model and proceeds with normal behavior.

**If the environment mismatches:**
  - The function raises an error, preventing the model from being logged.
  - The user must update their environment to match the requirements specified in python_env.yaml.
  - Or update the requirements.txt and python_env.yaml files to match the local environment.




### 2. load_registered_model and load_experiment_model

#### Purpose

The load_registered_model and load_experiment_model functions have been updated to enforce environment checks before loading a model. This ensures that the local active environment matches the environment logged with the model.

#### Key Features

- Verifies the local environment before loading a registered or experiment model.
- Prevents models from being loaded if the environment mismatches.

#### Usage

**No code changes are required. Continue using the functions as before:**

``` python
from base.v2.mlutils import load_registered_model

# Load a registered model
model_dict = mlutils.load_registered_model(
            model_name=MODEL_NAME,   # make sure it's matching the model name you set on mlflow server
            alias='best_model',  # make sure it's matching the alias you set on mlflow server
        )
```

#### Behavior

- If the environment matches:
- The model is loaded as usual.
- If the environment mismatches:
- An error is raised, preventing the model from being loaded.
- The user must update their local environment to match the requirements logged with the model.

#### Expected Benefits

**1. Improved Reproducibility:**
   - ures that models are logged and loaded in consistent environments, reducing dependency-related issues.
**2. Reliable Dependency Management:**
   - By enforcing environment checks, these functions help maintain compatibility between logged and active environments.
**3. Simplified Workflow:**
   - Developers can use talabat_log_model and the updated loading functions without modifying their existing code significantly. 
**4. Enhanced MLflow Dependency Inference:**
   - Encouraging the use of input_example during model logging improves MLflow’s ability to identify all necessary dependencies.

### Summary

#### Logging Models

- Use talabat_log_model to log models while ensuring environment consistency:
``` python
from base.v2.talabat_pyfunc import talabat_log_model
```

#### Loading Models

- Continue using load_registered_model or load_experiment_model to load models with enhanced environment checks:

``` python
from base.v2.mlutils import load_registered_model
```

### Environment Enforcement

In both cases:

- The local environment must match the environment logged with the model.
- Mismatches will prevent logging or loading, prompting users to update their environment.

By adopting these customized functions, teams can improve the reliability, reproducibility, and manageability of their MLflow workflows.