import sys
import os
directory_path = os.path.abspath("../../../../")
sys.path.append(directory_path)
print('Sys path', sys.path)
import mlflow
import structlog
from dotenv import load_dotenv
from base.v1 import mlutils
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import polars as pl
from typing import List
import argparse
from base.v0.perf import perf_manager


from projects.vendor_ranking.session_based_ranker.utils.data_utils import (
        read_data_fs
    )
from projects.vendor_ranking.session_based_ranker.utils.preprocess import (
    preprocess_evaluation,
    data_collator_without_label,
)

load_dotenv(override=True)
LOG: structlog.stdlib.BoundLogger = structlog.get_logger()

CHAIN_ID_KEY = 'chain_id'
LOGIT_KEY = 'logit'
BATCH_SIZE = 10000

def date_2_str(date_obj):
    return date_obj.strftime("%Y-%m-%d")

class Evaluate:

    def __init__(self, country):
        self.country = country
        self.MODEL_NAME = f"vendor_ranking_session_based_ranker_full_model_{self.country.lower()}"
        LOG.info(f"Tracking uri: {mlflow.get_tracking_uri()}")
        LOG.info(f"MODEL_NAME: {self.MODEL_NAME}")

        LOG.info("Retrieving the model from mlflow server...")
        model_dict = mlutils.load_registered_model(
            model_name=self.MODEL_NAME,
            alias='ace_champion_model_staging',
        )
        mlflow_model = model_dict['mlflow_model']
        self.model = mlflow_model.unwrap_python_model()

        # Extract model components
        self.recommender = self.model.recommender
        self.model_config = self.model.model_config
        self.area_id_to_index = self.model.area_id_to_index
        self.geohash_to_index = self.model.geohash_to_index
        self.chain_id_vocab = self.model.chain_id_vocab
        self.index_to_chain_id = self.model.index_to_chain_id
        self.chain_ids = self.model.chain_ids

        self.chain_ids_series = pd.Series(self.chain_ids, name=CHAIN_ID_KEY)
        self.chain_ids_set = set(self.chain_ids)

        self.model_config["combined_feature_configs"] = (self.model_config["feature_configs"] + self.model_config["online_feature_configs"])


    @staticmethod
    def prepare_chains_dataframe(predicted_logits: torch.Tensor, chain_ids_series) -> pl.LazyFrame:
        logits = predicted_logits.squeeze(0).tolist()
        return pl.LazyFrame(
            data={CHAIN_ID_KEY: chain_ids_series, LOGIT_KEY: logits},
            schema={CHAIN_ID_KEY: pl.Int32, LOGIT_KEY: pl.Float32},
            strict=False
        ).drop_nulls(subset=[CHAIN_ID_KEY])

    @staticmethod
    def sort_all_chains(chains_df: pl.LazyFrame, logit_col) -> List[int]:
        sorted_chains_df = chains_df.sort(
            logit_col, descending=True
        ).collect()
        sorted_chain = sorted_chains_df[CHAIN_ID_KEY].to_list()
        return sorted_chain

    def evaluate(self, test_df_path="test_df_session_ae.parquet", output_path="session_based_ranked_chains_ae.parquet"):
        # Load test data and get date range
        test_df = pd.read_parquet(test_df_path)
        start_date, end_date = date_2_str(test_df.order_date.min()), date_2_str(test_df.order_date.max())
        test_order_ids = list(test_df.order_id.unique())

        with perf_manager(
                description="Successfully loaded test order data",
        ):
            # Read feature data within the date range and filter by test order IDs
            order_features = read_data_fs(start_date, end_date, country_code=self.country)
            test_order_features = order_features[order_features.order_id.isin(test_order_ids)]
            print('Test order features', test_order_features.shape)
            print('Orders with no session clicks', test_order_features[test_order_features["session_clicks"] == "no_session_clicks"].shape)

        with perf_manager(
                description="Successfully tokenized and pre-processed the dataset",
        ):
            # Create a dataset and tokenize it
            test_order_dataset = Dataset.from_pandas(test_order_features)

            tokenized_test_dataset = test_order_dataset.map(
                lambda x: preprocess_evaluation(
                    x,
                    feature_configs=self.model_config["combined_feature_configs"],
                    area_id_to_index=self.area_id_to_index,
                    geohash_to_index=self.geohash_to_index,
                    numerical_feat_names=self.model_config["numerical_features"]
                ),
                batched=True
            )

            # Prepare dataloader for batched evaluation
            dataloader = DataLoader(
                tokenized_test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                collate_fn=lambda features: data_collator_without_label(
                    features, self.model_config["combined_feature_configs"]),
            )

        with perf_manager(
                description="Successfully completed the evaluation",
        ):
            not_found_errors = 0
            results = []

            for batch in tqdm(dataloader):
                # chain_id = batch[CHAIN_ID_KEY][0]

                chain_ids = batch[CHAIN_ID_KEY]
                order_ids = batch["order_id"]

                # Identify valid and invalid chain IDs
                valid_mask = np.array([chain_id in self.chain_ids_set for chain_id in chain_ids])
                invalid_indices = np.where(~valid_mask)[0]
                valid_indices = np.where(valid_mask)[0]

                with perf_manager(
                        description="Batch Inference",
                ):
                    with torch.no_grad():
                        _, predicted_logits = self.recommender(**batch)

                with perf_manager(
                        description="Batch Sorting",
                ):
                    # Handle invalid chains
                    for idx in invalid_indices:
                        results.append({
                            "order_id": order_ids[idx],
                            "rank": float('nan'),
                            "ranked_chains": float('nan')
                        })
                    not_found_errors += len(invalid_indices)

                    # Loop over just valid indices and fetch corresponding logits and chain_ids
                    for idx in valid_indices:
                        valid_chain_id = chain_ids[idx]
                        valid_logit = predicted_logits[idx]

                        # Prepare chains dataframe and sort for each order
                        chains_df = self.prepare_chains_dataframe(valid_logit, self.chain_ids_series)
                        sorted_chains = self.sort_all_chains(chains_df, LOGIT_KEY)

                        # Get the rank for the chain
                        rank = sorted_chains.index(int(valid_chain_id))  # Ensure you have a proper index

                        # Append the result
                        results.append({
                            "order_id": order_ids[idx],
                            "rank": rank,
                            "ranked_chains": sorted_chains
                        })

        with perf_manager(
                description="Successfully converted the results to parquet",
        ):
            results_df = pd.DataFrame(results)
            results_df.to_parquet(output_path)

        return results_df, not_found_errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluation for T5 model."
    )
    parser.add_argument(
        "--country",
        required=True,
        choices=['EG', 'AE', 'KW', 'QA', 'OM', 'BH', 'IQ', 'JO'],
        help="Possible country values are: ['EG', 'AE', 'KW', 'QA', 'OM', 'BH', 'IQ', 'JO']"
    )
    args = parser.parse_args()
    evaluate = Evaluate(country=args.country)
    evaluate.evaluate()