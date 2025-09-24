import sys
from pathlib import Path
root_dir = Path(__file__).parent
[sys.path.insert(0, str(root_dir := root_dir.parent)) for _ in range(3)]
import os
import mlflow
import structlog
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import cloudpickle
from base.v1.file_tools import collect_folders
from dotenv import load_dotenv
from model_wrapper import ModelWrapper


load_dotenv(override=True)
LOG: structlog.stdlib.BoundLogger = structlog.get_logger()


class ModelTrainer:
    def __init__(self):
        exp_name = os.environ["MLFLOW_EXPERIMENT_NAME"]
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
        pass


    def train(self, x_train, x_test, y_train, y_test):
        with mlflow.start_run() as run:  # Do not change this line

            # Training code here
            model = RandomForestClassifier(n_estimators=10)
            model.fit(x_train, y_train)
            accuracy = accuracy_score(y_test, model.predict(x_test))
            LOG.info(f"Model trained with accuracy: {accuracy}")

            # Save the model using mlflow's pyfunc
            current_dir = Path(__file__).parent
            model_path: Path = current_dir / f"{self.MODEL_NAME}.pkl"

            paths = [
                "projects/example/common",
                "projects/example/iris_classifier_test",
                "base",
            ]
            artifacts = {
                "model": model_path,
                # add more artifacts as needed
            }

            with collect_folders(
                    root=root_dir,
                    paths=paths,
                    temp_artifact_file_paths=[model_path]
            ) as tmp_dir:

                model_path.write_bytes(cloudpickle.dumps(model))

                # Save the model using mlflow's pyfunc.
                # The mlflow.pyfunc.log_model() method is used to log the model and the code paths,
                # and other artifacts.
                mlflow.pyfunc.log_model(
                    artifact_path='model',
                    python_model=ModelWrapper(),
                    artifacts={k: str(v) for k, v in artifacts.items()},
                    # paths to folders to be logged into MLFlow
                    code_path=[
                        tmp_dir / "projects",
                        tmp_dir / "base"
                    ],
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
        x_train, x_test, y_train, y_test = self.get_data()
        self.train(x_train, x_test, y_train, y_test)




if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer()
