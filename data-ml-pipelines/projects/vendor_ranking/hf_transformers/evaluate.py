import mlflow
import structlog
from dotenv import load_dotenv
from base.v1 import mlutils
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import polars as pl
from typing import List
import argparse


from projects.vendor_ranking.hf_transformers.utils.data_utils import (
        read_data_fs
    )
from projects.vendor_ranking.hf_transformers.utils.preprocess import (
    preprocess_evaluation,
    data_collator_without_label,
)

load_dotenv(override=True)
LOG: structlog.stdlib.BoundLogger = structlog.get_logger()

CHAIN_ID_KEY = 'chain_id'
LOGIT_KEY = 'logit'
BATCH_SIZE = 1

def date_2_str(date_obj):
    return date_obj.strftime("%Y-%m-%d")

class Evaluate:

    def __init__(self, country):
        self.country = country
        self.MODEL_NAME = f"vendor_ranking_hf_transformers_full_model_{self.country.lower()}"
        LOG.info(f"Tracking uri: {mlflow.get_tracking_uri()}")
        LOG.info(f"MODEL_NAME: {self.MODEL_NAME}")

        LOG.info("Retrieving the model from mlflow server...")
        model_dict = mlutils.load_registered_model(
            model_name=self.MODEL_NAME,
            alias='best_full_model',
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

    def evaluate(self, test_df_path="test_df.parquet"):
        # Load test data and get date range
        test_df = pd.read_parquet(test_df_path)
        start_date, end_date = date_2_str(test_df.order_date.min()), date_2_str(test_df.order_date.max())
        test_order_ids = list(test_df.order_id.unique())

        # Read feature data within the date range and filter by test order IDs
        order_features = read_data_fs(start_date, end_date, country_code=self.country)
        test_order_features = order_features[order_features.order_id.isin(test_order_ids)]

        # Create a dataset and tokenize it
        test_order_dataset = Dataset.from_pandas(test_order_features)

        tokenized_test_dataset = test_order_dataset.map(
            lambda x: preprocess_evaluation(
                x,
                feature_configs=self.model_config["feature_configs"],
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
            collate_fn=lambda features: data_collator_without_label(features, self.model_config["feature_configs"]),
        )

        not_found_errors = 0
        results = []

        for batch in tqdm(dataloader):
            chain_id = batch[CHAIN_ID_KEY][0]

            # If chain_id is not in the set, mark rank and ranked_chains as NaN
            if chain_id not in self.chain_ids_set:
                batch["rank"] = float('nan')
                batch["ranked_chains"] = float('nan')
                not_found_errors += 1

                results.append(
                    {
                        "order_id": batch["order_id"][0],
                        "rank": batch["rank"],
                        "ranked_chains": batch["ranked_chains"]
                    })
                continue


            with torch.no_grad():
                _, predicted_logits = self.recommender(**batch)

            # Prepare and sort chains by predicted logits
            chains_df = self.prepare_chains_dataframe(predicted_logits, self.chain_ids_series)
            sorted_chains = self.sort_all_chains(chains_df, LOGIT_KEY)

            rank = sorted_chains.index(int(chain_id))
            batch["rank"] = rank
            batch["ranked_chains"] = sorted_chains

            results.append(
                {
                    "order_id": batch["order_id"][0],
                    "rank": batch["rank"],
                    "ranked_chains": batch["ranked_chains"]
                })

        results_df = pd.DataFrame(results)
        results_df.to_parquet('t5_sorted_chains.parquet')
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

