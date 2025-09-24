import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List

import mlflow
import structlog
import torch
import polars as pl

root_dir = Path(__file__).parent
[sys.path.insert(0, str(root_dir := root_dir.parent)) for _ in range(3)]
from base.v0.mlclass import MlflowBase
from base.v0.ml_metadata import find_model_by_metadata
from base.v0.perf import perf_manager
import newrelic.agent

from projects.vendor_ranking.hf_transformers.wrapper_utils.load_utils import (
    load_offline_model,
    load_online_model,
    load_model,
    load_feature_tokenizers
)

LOG: "structlog.stdlib.BoundLogger" = structlog.getLogger(__name__)

REQUEST_CHAINS_KEY = 'request_chains'
CHAIN_ID_KEY = 'chain_id'
LOGIT_KEY = 'logit'
REQUEST_ORDER_KEY = "request_order"


class ModelWrapper(MlflowBase):
    def __init__(self) -> None:
        self.offline_model = None
        self.online_model = None
        self.recommender = None
        self.model_config = None
        self.area_id_to_index = None
        self.geohash_to_index = None
        self.chain_id_vocab = None
        self.index_to_chain_id = None
        self.device = "cpu"

        self.transaction_name: Optional[str] = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        with perf_manager(
                description="Successfully loaded the model",
                description_before=f"-- Loading model context {context.artifacts} --"
        ):
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

            self.model_config_path = context.artifacts["model_config"]

            with open(self.model_config_path, 'r') as f:
                self.model_config = json.load(f)

            self.offline_model = load_offline_model(
                self.model_config,
                context.artifacts["offline_model"]
            )

            self.online_model, self.area_id_to_index, self.geohash_to_index, self.chain_id_vocab = load_online_model(
                self.model_config,
                context.artifacts["online_model"],
                context.artifacts["area_id_to_index"],
                context.artifacts["geohash_to_index"],
                context.artifacts["chain_id_vocab"]
            )

            self.recommender, self.area_id_to_index, self.geohash_to_index, self.chain_id_vocab = load_model(
                self.model_config,
                context.artifacts["offline_model"],
                context.artifacts["online_model"],
                context.artifacts["area_id_to_index"],
                context.artifacts["geohash_to_index"],
                context.artifacts["chain_id_vocab"]
            )

            load_feature_tokenizers(
                self.model_config['feature_configs'],
                context.artifacts['tokenizers']
            )

            self.offline_model = self.offline_model.to(self.device)
            self.offline_model.eval()
            self.online_model = self.online_model.to(self.device)
            self.online_model.eval()
            self.recommender = self.recommender.to(self.device)
            self.recommender.eval()

            self.index_to_chain_id = {v: k for k, v in self.chain_id_vocab.items()}

            self.total_chain_ids = len(self.chain_id_vocab)
            self.chain_ids = [self.index_to_chain_id[i] for i in range(self.total_chain_ids)]
            self.chain_ids_series = pl.Series(CHAIN_ID_KEY, self.chain_ids)

            if new_relic_app_name := os.environ.get("NEW_RELIC_APP_NAME"):
                newrelic_config_file = Path(__file__).parent / "newrelic-agent.ini"
                assert newrelic_config_file.exists(), f"New Relic configuration file not found: {newrelic_config_file}"
                newrelic.agent.initialize(config_file=str(newrelic_config_file))
                newrelic.agent.register_application()

                LOG.info(
                    f"New Relic agent initialized: "
                    f"config file = {newrelic_config_file}, app name = {new_relic_app_name}"
                )
            else:
                LOG.warning(
                    f"New Relic agent IS NOT initialized. Env var `NEW_RELIC_APP_NAME` not found"
                )

            self.model_info, self.model_version = find_model_by_metadata()
            if self.model_info and self.model_version:
                LOG.info(f"Model name: {self.model_info.name}, version: {self.model_version.version}")
                self.transaction_name = f"{self.model_info.name}:{self.model_version.version}"

            self.response = {
                "version": self.model_version and self.model_version.version,
                "model": self.model_info and self.model_info.name
            }

    @classmethod
    def validate_request(cls, request: dict[str, list[Any]]) -> dict[str, Any]:
        return {key: value[0] for key, value in request.items()}  # get the first values

    @newrelic.agent.function_trace()
    @staticmethod
    def preprocess_input(model_input, feature_configs, area_id_to_index, geohash_to_index, numerical_feature_names):
        for feature_config in feature_configs:
            name = feature_config['name']
            tokenizer = feature_config['tokenizer']
            max_length = feature_config['max_length']

            inputs = tokenizer(model_input[name], max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
            model_input[f'{name}_ids'] = inputs['input_ids']#.to(device)
            model_input[f'{name}_mask'] = inputs['attention_mask']#.to(device)

        model_input['order_hour'] = torch.tensor([model_input['order_hour']], dtype=torch.long)#.to(device)
        model_input['delivery_area_id'] = torch.tensor([area_id_to_index.get(model_input['delivery_area_id'], area_id_to_index['unk'])],
                                               dtype=torch.long)  # .to(device)
        model_input['geohash6'] = torch.tensor([geohash_to_index.get(model_input['geohash6'], geohash_to_index['unk'])],
                                               dtype=torch.long)  # .to(device)
        model_input['numerical_features'] = torch.tensor([model_input[feat] for feat in numerical_feature_names],
                                                                dtype=torch.float).T.unsqueeze(0)
        return model_input

    @newrelic.agent.function_trace()
    def inference(self, model_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            _, predicted_logits = self.recommender(**model_input)
        return predicted_logits

    @newrelic.agent.function_trace()
    @staticmethod
    def prepare_chains_dataframe(predicted_logits: torch.Tensor, chain_ids_series) -> pl.LazyFrame:
        logits = predicted_logits.squeeze(0).tolist()
        return pl.LazyFrame(
            data={CHAIN_ID_KEY: chain_ids_series, LOGIT_KEY: logits},
            schema={CHAIN_ID_KEY: pl.Int32, LOGIT_KEY: pl.Float32},
            strict=False
        ).drop_nulls(subset=[CHAIN_ID_KEY])

    @newrelic.agent.function_trace()
    @staticmethod
    def sort_all_chains(chains_df: pl.LazyFrame, logit_col) -> Dict[str, List]:
        sorted_chains_df = chains_df.sort(
            logit_col, descending=True
        ).collect()
        return sorted_chains_df.to_dict(as_series=False)

    @newrelic.agent.function_trace()
    @staticmethod
    def filter_and_sort_request_chains(chains_df: pl.LazyFrame, request_chains: List[int], logit_col) -> Dict[str, List]:
        request_chains_df = pl.LazyFrame(
            data={CHAIN_ID_KEY: request_chains, REQUEST_ORDER_KEY: range(len(request_chains))},
            schema={CHAIN_ID_KEY: pl.Int32, REQUEST_ORDER_KEY: pl.UInt32},
            strict=True
        )

        intersection_df = request_chains_df.join(other=chains_df, on=CHAIN_ID_KEY, how='left')

        sorted_df: pl.DataFrame = intersection_df.sort(
            by=(logit_col, pl.col(REQUEST_ORDER_KEY)),
            descending=True,
            nulls_last=True,
            maintain_order=True
        ).collect()

        return sorted_df.to_dict(as_series=False)


    @newrelic.agent.web_transaction()
    def predict(self, context: mlflow.pyfunc.PythonModelContext, request: dict[str, list[Any]]):

        newrelic.agent.set_transaction_name(self.transaction_name or "VR_T5_Full")

        model_input: dict[str, Any] = self.validate_request(request)
        request_chains: Optional[list[int]] = model_input.get(REQUEST_CHAINS_KEY)
        logit_col = pl.col(LOGIT_KEY)

        # 'preprocess' step
        model_input = self.preprocess_input(
            model_input,
            self.model_config["feature_configs"],
            self.area_id_to_index,
            self.geohash_to_index,
            self.model_config["numerical_features"]
        )

        # 'inference' step
        predicted_logits = self.inference(model_input)

        # Prepare chains dataframe
        chains_df = self.prepare_chains_dataframe(predicted_logits, self.chain_ids_series)

        if request_chains is None:
            # Sort all chains
            sorted_chains_df = self.sort_all_chains(chains_df, logit_col)
            return {
                "value": [sorted_chains_df[CHAIN_ID_KEY], sorted_chains_df[LOGIT_KEY]]
            } | self.response

        # Filter and sort chains based on requested chains
        sorted_chains_df = self.filter_and_sort_request_chains(chains_df, request_chains, logit_col)

        return {
            "value": [sorted_chains_df[CHAIN_ID_KEY], sorted_chains_df[LOGIT_KEY]]
        } | self.response


    @staticmethod
    def get_sample_input() -> Dict:
        model_input = {
            'delivery_area_id': ['1310'],
            'order_hour': [15],
            'geohash6': ['thrq84'],
            'prev_chains': ['637986 637986 637986 637986 637986 657351 665509 665509 665509 20758'],
            'freq_chains': ['665509 637986 638813 661836 656374 601837 647199 657351'],
            'prev_clicks': ['676148 609618 625259 679404 662242 678528 676196 667899 17121 673466'],
            'freq_clicks': ['649659 22290 1106 672025 7336 9776 1329 641623 677744 646617 28612 669785 677440 644562 609618'],
            'frequent_neg_impressions': ['no_negative_impressions'],
            'account_log_order_cnt': [3.6888794541139363],
            'account_log_avg_gmv_eur': [3.4722462],
            'account_incentives_pct': [0.3846153846153846],
            'account_is_tpro': [0.0],
            'account_discovery_pct': [0.5641025641025641],
            'request_chains': [None]
        }
        return model_input
