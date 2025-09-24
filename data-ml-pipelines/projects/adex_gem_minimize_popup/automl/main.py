import sys
from pathlib import Path
root_dir = Path(__file__).parent
[sys.path.insert(0, str(root_dir := root_dir.parent)) for _ in range(3)]

import numpy as np

# Temporarily replace np.object with object to avoid deprecation errors
try:
    np.object = object
except AttributeError:
    pass
from model_wrapper import ModelWrapper
from base.v0.file_tools import collect_folders
import mlflow
import os
import structlog
from pathlib import Path
from dotenv import load_dotenv
from base.v1.mlutils import register_model


load_dotenv(override=True)
LOG: structlog.stdlib.BoundLogger = structlog.get_logger()
REPO_ROOT_PATH = Path(__file__).parent.parent.parent.parent


class ModelTrainer:
    def __init__(self):
        exp_name = os.environ["MLFLOW_EXPERIMENT_NAME"]
        mlflow.set_experiment(exp_name)
        self.MODEL_NAME = exp_name
        LOG.info(f"Tracking uri: {mlflow.get_tracking_uri()}")
        LOG.info(f"Experiment id: {exp_name}")


    def automl_to_mlflow(self):

        current_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(current_path, '/Users/fadi.baskharon/Downloads/model/predict/001')

        with mlflow.start_run() as run:  # Do not change this line

            # paths (relative to `root_path`) to folders which must be copied
            paths = [
                "projects/adex_gem_minimize_popup/common",
                "projects/adex_gem_minimize_popup/automl",
                "base",
            ]

            with collect_folders(root=REPO_ROOT_PATH, paths=paths) as tmp_dir:
                # Save the model using mlflow's pyfunc.
                # The mlflow.pyfunc.log_model() method is used to log the model and the code paths,
                # and other artifacts.
                LOG.info(f"Logging model to MLFlow")
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=ModelWrapper(),
                    artifacts={
                        "model": model_path,
                        # add more artifacts as needed
                    },
                    code_path=[
                        tmp_dir / "projects",
                        tmp_dir / "base"
                    ],

                )
                LOG.info(f"Model logged to MLFlow")

            # register the model
            register_model(
                model_name=self.MODEL_NAME,
                run=run,
                alias="best_model",
                tags={
                },
            )


    def __call__(self):
        # the main logic of the class, where you call the methods in the needed order
        self.automl_to_mlflow()



if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer()
