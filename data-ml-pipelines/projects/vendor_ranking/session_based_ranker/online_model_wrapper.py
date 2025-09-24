import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List
import mlflow
import numpy as np
import structlog
import torch
import polars as pl
from pydantic import Field
from pydantic_settings import BaseSettings

root_dir = Path(__file__).parent
[sys.path.insert(0, str(root_dir := root_dir.parent)) for _ in range(3)]
from base.v0.mlclass import MlflowBase
from base.v0.ml_metadata import find_model_by_metadata
from base.v0.perf import perf_manager
import newrelic.agent

from projects.vendor_ranking.session_based_ranker.wrapper_utils.load_utils import (
    load_online_model,
    load_feature_tokenizers
)

LOG: "structlog.stdlib.BoundLogger" = structlog.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


REQUEST_CHAINS_KEY = 'request_chains'
CHAIN_ID_KEY = 'chain_id'
LOGIT_KEY = 'logit'
REQUEST_ORDER_KEY = "request_order"


class InferenceParams(BaseSettings):
    torch_num_threads: Optional[int] = Field(default=None, env="TORCH_NUM_THREADS", gt=0)

# Load inference parameters
INFERENCE_PARAMS = InferenceParams()
TORCH_NUM_THREADS = INFERENCE_PARAMS.torch_num_threads

# Apply thread settings only if explicitly set
if TORCH_NUM_THREADS is not None:
    torch.set_num_threads(TORCH_NUM_THREADS)
    torch.set_num_interop_threads(TORCH_NUM_THREADS)
    LOG.info(f"PyTorch computation threads set to : {TORCH_NUM_THREADS}")
else:
    LOG.info("TORCH_NUM_THREADS not set. Using PyTorch default thread settings.")

LOG.info(f"PyTorch computation threads: {torch.get_num_threads()}")


class OnlineModelWrapper(MlflowBase):
    def __init__(self) -> None:
        self.online_model = None
        self.model_config = None
        self.area_id_to_index = None
        self.geohash_to_index = None
        self.chain_id_vocab = None
        self.index_to_chain_id = None
        self.device = torch.device('cpu') #"cpu"

        self.transaction_name: Optional[str] = None
        

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        with perf_manager(
                description="Successfully loaded the online model",
                description_before=f"-- Loading model context {context.artifacts} --"
        ):
            self.device = torch.device("cpu")

            self.model_config_path = context.artifacts["model_config"]

            with open(self.model_config_path, 'r') as f:
                self.model_config = json.load(f)

            self.online_model, self.area_id_to_index, self.geohash_to_index, self.chain_id_vocab = load_online_model(
                self.model_config,
                context.artifacts["online_model"],
                context.artifacts["area_id_to_index"],
                context.artifacts["geohash_to_index"],
                context.artifacts["chain_id_vocab"]
            )

            load_feature_tokenizers(
                self.model_config['online_feature_configs'],
                context.artifacts['tokenizers']
            )

            self.online_model = self.online_model.to(self.device)
            self.online_model.set_model_device(self.device)
            self.online_model.eval()

            # Create the mapping from tokenizer's vocabulary
            self.index_to_chain_id = {v: k for k, v in self.chain_id_vocab.items()}

            self.total_chain_ids = len(self.chain_id_vocab)
            self.chain_ids = [self.index_to_chain_id[i] for i in range(self.total_chain_ids)]
            self.chain_ids_series = pl.Series(CHAIN_ID_KEY, self.chain_ids)

            with perf_manager(
                    description="Successfully warmed up the online model"
            ):
                self.model_warmup(self.model_config["offline_hidden_size"])

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

    def model_warmup(self, embedding_length: int):
        offline_embedding = np.random.rand(embedding_length)

        model_input = {
            'offline_embedding': offline_embedding,
            'order_hour': 23,
            'delivery_area_id': '756999',
            'geohash6': 'thrqb5',
            'session_clicks': ["test"]
        }
        model_input = self.preprocess_input(
            model_input,
            self.model_config["online_feature_configs"],
            self.area_id_to_index,
            self.geohash_to_index,
            self.device
        )

        for _ in range(10):
            self.inference(model_input)


    ### Maybe update this function to move data to device during preprocessing ? 
    @newrelic.agent.function_trace()
    @staticmethod
    def preprocess_input(model_input, feature_configs, area_id_to_index, geohash_to_index, device):
        float32_array = model_input['offline_embedding'].astype(np.float32)
        model_input['offline_embedding'] = float32_array.tolist()
        model_input['offline_embedding'] = torch.tensor([model_input['offline_embedding']], device=device)

        tokenized_inputs = {}
        for feature_config in feature_configs:
            name = feature_config['name']
            tokenizer = feature_config['tokenizer']
            max_length = feature_config['max_length']

            inputs = tokenizer(model_input[name], max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
            # model_input[f'{name}_ids'] = inputs['input_ids']
            # model_input[f'{name}_mask'] = inputs['attention_mask']

            tokenized_inputs[f'{name}_ids'] = inputs['input_ids']
            tokenized_inputs[f'{name}_mask'] = inputs['attention_mask']

        model_input['tokenized_inputs'] = tokenized_inputs

        model_input['order_hour'] = torch.tensor([model_input['order_hour']], dtype=torch.long)
        model_input['delivery_area_id'] = torch.tensor(
            [area_id_to_index.get(model_input['delivery_area_id'], area_id_to_index['unk'])],
            dtype=torch.long)
        model_input['geohash6'] = torch.tensor([geohash_to_index.get(model_input['geohash6'], geohash_to_index['unk'])],
                                               dtype=torch.long)
        return model_input

    @newrelic.agent.function_trace()
    def inference(self, model_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            _, predicted_logits = self.online_model(
                model_input['offline_embedding'],
                model_input['order_hour'],
                model_input['delivery_area_id'],
                model_input['geohash6'],
                labels=None,
                **model_input['tokenized_inputs']
            )
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
        with perf_manager(
                description="Time taken for 'inference server'"
        ):
            newrelic.agent.set_transaction_name(self.transaction_name or "VR_Session_Based")

            model_input: dict[str, Any] = self.validate_request(request)
            request_chains: Optional[list[int]] = model_input.get(REQUEST_CHAINS_KEY)
            logit_col = pl.col(LOGIT_KEY)

            #TODO: check to reject if no session clicks

            # 'preprocess' step
            model_input = self.preprocess_input(
                model_input,
                self.model_config["online_feature_configs"],
                self.area_id_to_index,
                self.geohash_to_index,
                self.device
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
            'offline_embedding': [[0.1484894, 1.8863342, 0.89126116, 0.32214823, 0.06425029, -1.2298694, -0.21821447]],
            # has to be length of embedding
            'order_hour': [20],
            'delivery_area_id': ['1192'],
            'geohash6': ['thrqb5'],
            'session_clicks': ['677744 646617 28612 669785 677440 644562 609618'],
            'request_chains': [None]
        }
        return model_input
