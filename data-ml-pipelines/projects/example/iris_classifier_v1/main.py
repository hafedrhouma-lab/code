import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING
import cloudpickle
import mlflow
import structlog
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from model_wrapper import ModelWrapper
from base.v2.file_tools import collect_files, temp_open_file, temp_dir
from base.v2.talabat_pyfunc import talabat_log_model
from base.v0.file_tools import collect_folders
from base.v2.mlutils import register_model, load_registered_model


from dotenv import load_dotenv
load_dotenv(override=True)
warnings.simplefilter(action='ignore', category=FutureWarning)

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo

# current file full path of the parent directory
from pathlib import Path
FILE_PATH = Path(__file__).parent



LOG: structlog.stdlib.BoundLogger = structlog.get_logger()

class ModelTrainer:
    def __init__(self):
        exp_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        self.MODEL_NAME = exp_name
        self.alias = "best_model"
        LOG.info(f"Tracking uri: {mlflow.get_tracking_uri()}")
        LOG.info(f"Experiment id: {exp_name}")

    import mlflow
    import sys
    import yaml
    from mlflow.models import infer_pip_requirements
    from pathlib import Path
    from typing import Optional, Union

    def train(self, n_estimators=100, max_depth=2, criterion="gini"):
        # Example training code
        iris = load_iris()
        x_train, x_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42
        )
        model = RandomForestClassifier()
        model.fit(x_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(x_test))

        # Log training details with MLFlow
        model_params = {
            "model_type": "RandomForest",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "criterion": criterion
        }
        with mlflow.start_run() as self.run:
            mlflow.log_params(model_params)
            mlflow.log_metric("accuracy", model.score(x_test, y_test))

            # Create a temporary file to save the serialized model
            BASE_PATH = Path(__file__).parent
            root_path: Path = BASE_PATH.parent.parent.parent
            model_path: Path = BASE_PATH / f"{self.MODEL_NAME}.pkl"
            artifacts = {
                "model": model_path,
            }

            with temp_open_file(model_path, "wb") as f:
                f.write(cloudpickle.dumps(model))

                # Save the model using mlflow's pyfunc

                with collect_folders(
                        root=root_path,
                        paths=[
                            "projects/example/iris_classifier_v1",
                            "base",
                        ]
                ) as tmp_dir:


                    model_info: "ModelInfo" = talabat_log_model(
                        artifact_path="model",
                        python_model=ModelWrapper(),
                        artifacts={k: str(v) for k, v in artifacts.items()},
                        code_paths=[
                            tmp_dir / "projects",
                            tmp_dir / "base"
                        ],
                        pip_requirements=f"{FILE_PATH}/requirements.txt",
                        input_example=[5.1, 3.5, 1.4, 0.2]
                    )

                LOG.info(
                    f"Model: `{self.MODEL_NAME}` logged to mlflow server. "
                    f"Run URI: {model_info}. "
                    f"Params {model_params}."
                )
                return model_info

    def register_model(self):
        # register the model
        register_model(
            model_name=self.MODEL_NAME,
            run=self.run,
            alias=self.alias,
            tags={"project": "dependency-management"
                  },
        )

    def load_registered_model(self):
        # Load the registered model
        model = load_registered_model(self.MODEL_NAME, self.alias)
        LOG.info(f"Model loaded from MLFlow: {model}")

    def __call__(self):
        self.train()

        
if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer()
    LOG.info("Model trained successfully")
    trainer.register_model()
    LOG.info("Model registered successfully")
    trainer.load_registered_model()
    LOG.info("Model loaded successfully")


