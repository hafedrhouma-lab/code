import os
import mlflow
import structlog
from pathlib import Path
from base.v0.file_tools import collect_files, temp_open_file, temp_dir
from dotenv import load_dotenv
from model_wrapper import ModelWrapper
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import cloudpickle
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
        LOG.info("Loading data")
        iris = load_iris()
        x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
        return x_train, x_test, y_train, y_test

    def preprocess_data(self):
        # Data preprocessing code
        LOG.info(("No preprocessing needed"))

    def train(self, x_train, x_test, y_train, y_test):

        with mlflow.start_run() as run:  # Do not change this line

            # Training code here
            model = RandomForestClassifier(n_estimators=10)
            model.fit(x_train, y_train)
            accuracy = accuracy_score(y_test, model.predict(x_test))
            LOG.info(f"Model trained with accuracy: {accuracy}")


            # Save the model using mlflow's pyfunc
            model_pkl_file = f"{self.MODEL_NAME}.pkl"
            with temp_open_file(model_pkl_file, "wb") as f:
                f.write(cloudpickle.dumps(model))

                ## The code_paths variable is used to specify the paths to the files that are needed to load the model
                code_paths = collect_files(
                    base=Path(__file__).parent,
                    src_dirs=["../../../base", "../common"],
                    src_files=["model_wrapper.py"]
                )
                with temp_dir(base="./", paths=code_paths) as dir_path:
                    ## The mlflow.pyfunc.log_model() method is used to log the model and the code paths, and other artifacts
                    mlflow.pyfunc.log_model(
                        artifact_path='model',
                        python_model=ModelWrapper(),
                        artifacts={
                            "model": model_pkl_file,
                            # add more artifacts as needed
                        },
                        code_path=code_paths
                    )

            # Log the parameters and metrics - Report the actual values
            mlflow.log_params(
                model.get_params()
            )
            mlflow.log_metrics(
                {
                    "accuracy": accuracy,

                }
            )
            return

    def __call__(self):
        # the main logic of the class, where you call the methods in the needed order
        x_train, x_test, y_train, y_test = self.get_data()
        self.train(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer()
