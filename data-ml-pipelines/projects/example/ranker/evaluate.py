import warnings
import structlog
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from projects.example.common.abstract_evaluator import AbstractEvaluator


load_dotenv(override=True)
warnings.simplefilter(action='ignore', category=FutureWarning)
LOG: structlog.stdlib.BoundLogger = structlog.get_logger()


class ModelEvaluator(AbstractEvaluator):
    def __init__(self):
        super().__init__()

    def evaluate(self):
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
        # Generate a toy dataset based on the user's description
        num_samples = 1000  # Number of orders
        num_chains_per_order = 30  # Number of chains to rank per order

        # Generate order IDs
        order_ids = np.arange(num_samples)
        chain_rank= np.random.randint(1, num_chains_per_order + 1, num_samples)
        # Combine into a DataFrame, simulating the clicked chain's rank per order
        df_ranking = pd.DataFrame({
            'order_id': order_ids,
            'chain_rank': chain_rank
        })

        return df_ranking

    def __call__(self):
        df_ranking = self.evaluate()
        self.log_evaluation(df_ranking)


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator()
