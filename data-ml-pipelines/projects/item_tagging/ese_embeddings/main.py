import sys
from pathlib import Path
root_dir = Path(__file__).parent
[sys.path.insert(0, str(root_dir := root_dir.parent)) for _ in range(3)]
import os
import mlflow
import structlog
from pathlib import Path
from base.v0.file_tools import collect_files, temp_open_file, temp_dir
from dotenv import load_dotenv
from model_wrapper import ModelWrapper

load_dotenv(override=True)
LOG: structlog.stdlib.BoundLogger = structlog.get_logger()


class ModelTrainer:
    def __init__(self):
        exp_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        self.MODEL_NAME = exp_name
        LOG.info(f"Tracking uri: {mlflow.get_tracking_uri()}")
        LOG.info(f"Experiment id: {exp_name}")

    def get_data(self):
        # Data loading code
        pass

    def preprocess_data(self):
        # Data preprocessing code
        pass

    def train(self):

        with mlflow.start_run() as run:  # Do not change this line

            # Training code here


            # Save the model using mlflow's pyfunc

            ## The code_paths variable is used to specify the paths to the files that are needed to load the model
            code_paths = collect_files(
                base=Path(__file__).parent,
                src_dirs=["../../../base", "../common"],
                src_files=["model_wrapper.py"]
            )
            ## The mlflow.pyfunc.log_model() method is used to log the model and the code paths, and other artifacts
            mlflow.pyfunc.log_model(
                artifact_path='model',
                python_model=ModelWrapper(),
                artifacts={
                    # "model": f"{self.MODEL_NAME}.pkl",
                    # add more artifacts as needed
                },
                code_path=code_paths
            )

            # Log the parameters and metrics - Report the actual values
            mlflow.log_params(
                {
                    "param_1": 0.5,
                    "param_2": 0.6,
                    "param_3": 0.7,
                }
            )
            mlflow.log_metrics(
                {
                    "metric_1": 0.8,
                    "metric_2": 0.85,
                    "metric_3": 0.9,
                }
            )

    def __call__(self):
        # the main logic of the class, where you call the methods in the needed order
        self.get_data()
        self.preprocess_data()
        self.train()



if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer()
