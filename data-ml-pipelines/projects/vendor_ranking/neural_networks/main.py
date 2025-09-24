import os
from model_wrapper import ModelWrapper
import structlog
import mlflow
from dotenv import load_dotenv
load_dotenv()


LOG: structlog.stdlib.BoundLogger = structlog.get_logger()


class ModelTrainer:
    def __init__(self):
        exp_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        self.MODEL_NAME = exp_name
        LOG.info(f"Tracking uri: {mlflow.get_tracking_uri()}")
        LOG.info(f"Experiment id: {exp_name}")

    # 1. You can create as many methods as needed to train your model
    # 2. The train method is where you train and track your experiments
    #    >>> You should not change this line `with mlflow.start_run() as self.run:`
    # 3. Finally, define the execution logic in the __call__ method

    def train(self):
        # Training code
        # pass

        # Log training details with MLflow
        with mlflow.start_run() as run:  # Do not change this line
            mlflow.log_params({
                "model_type": "Neural Network",
                "learning_rate": 0.01,
                "epochs": 100
            }
        )

    def __call__(self):
        self.train()


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer()
