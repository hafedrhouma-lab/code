import os
import warnings
import mlflow
import structlog
import pandas as pd
from dotenv import load_dotenv
from projects.example.common.metrics import recall_10
from abc import ABC, abstractmethod


load_dotenv(override=True)
warnings.simplefilter(action='ignore', category=FutureWarning)
LOG: structlog.stdlib.BoundLogger = structlog.get_logger()


class AbstractEvaluator(ABC):
    def __init__(self):
        exp_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        self.MODEL_NAME = exp_name
        LOG.info(f"Tracking uri: {mlflow.get_tracking_uri()}")
        LOG.info(f"Experiment id: {exp_name}")

    @abstractmethod
    def evaluate(self) -> pd.DataFrame:
        """
        This method should be implemented by the subclass to evaluate
        Returns:
            pd.DataFrame: A DataFrame containing the evaluation data in the following format:
                order_id | chain_rank
                0        | 11
                1        | 9
                2        | 5
                ...      | ...
        """
        pass

    @staticmethod
    def log_evaluation(df_ranking):
        with mlflow.start_run(run_name="evaluation"):
            # Define the custom metric for MLflow
            recall_10_metric = mlflow.models.make_metric(
                eval_fn=recall_10,
                greater_is_better=True,
            )

            # Simulated evaluation data in DataFrame format (replace this with the actual DataFrame if necessary)
            eval_df = df_ranking.copy()

            # Evaluate the dataset on MLflow, using only predictions as per your dataset evaluation requirement
            results = mlflow.evaluate(
                data=eval_df,
                predictions="chain_rank",
                evaluators="default",
                extra_metrics=[recall_10_metric]
            )

            LOG.info(f"Evaluation metrics: {results.metrics}")

            # Log evaluation data as a table artifact in JSON format
            mlflow.log_table(data=eval_df, artifact_file="evaluation_data.json")

    def __call__(self):
        df_ranking = self.evaluate()
        self.log_evaluation(df_ranking)
