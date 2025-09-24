from pathlib import Path

import newrelic.agent
import numpy as np
import pandas as pd
import structlog
import tensorflow as tf

from base.v0.perf import perf_manager
from . import USER_MODEL_INPUT, CHAIN_MODEL_INPUT, WARMUP_USER_MODEL_INPUT_2, WARMUP_USER_MODEL_INPUT
from ..data.datasets.tf_data import (
    create_tf_dataset_from_tensor_slices
)
from ..model.ese_embedding_layer import get_ese_embedding_tf_layer
from ..model.two_tower_model import create_two_tower_model
from ..utils.load_utils import load_model_config

LOG: "structlog.stdlib.BoundLogger" = structlog.getLogger(__name__)


DTYPE_MAPPING = {
    int: tf.int64,
    float: tf.float64,
    str: tf.string,
    np.int64: tf.int64,
    np.float64: tf.float64,
    np.str_: tf.string,
}


class TwoTowersPredictor:
    def __init__(
            self,
            params_input_data,
            model_config_path: Path,
            ese_chain_embeddings_path: str,
            user_weights_file,
            chain_weights_file
    ):
        self.params = load_model_config(path=model_config_path)
        self.params_input_data = params_input_data
        self.user_model_input = USER_MODEL_INPUT
        self.user_model_input_2 = WARMUP_USER_MODEL_INPUT_2
        self.user_model_input_3 = WARMUP_USER_MODEL_INPUT
        self.chain_model_input = CHAIN_MODEL_INPUT
        self.user_weights_file = user_weights_file
        self.chain_weights_file = chain_weights_file

        self.ese_chain_embeddings = pd.read_parquet(
            ese_chain_embeddings_path,
            engine="pyarrow"
        )

        candidates_ds = create_tf_dataset_from_tensor_slices(
            self.params_input_data.chain_features_df
        )

        ese_vec_embedding_layer = get_ese_embedding_tf_layer(
            is_vector=True,
            embeddings_df=self.ese_chain_embeddings
        )
        ese_ch_embedding_layer = get_ese_embedding_tf_layer(
            is_vector=False,
            embeddings_df=self.ese_chain_embeddings
        )
        self.model = create_two_tower_model(
            embedding_dimension=self.params.get("embedding_dimension"),
            unique_customer_ids=self.params.get("unique_customer_ids"),
            unique_chain_ids=self.params_input_data.unique_chain_ids,
            unique_cuisines_names=self.params_input_data.unique_cuisines_names,
            candidates_ds=candidates_ds,
            prev_chains_feat=True,
            enable_prev_searches=self.params.get("enable_prev_searches"),
            enable_prev_items=self.params.get("enable_prev_items"),
            items_vocab=self.params.get("items_vocab_list"),
            searches_vocab=self.params_input_data.search_vocab_list,
            chain_cuisine_feat=True,
            handle_popularity_bias=True,
            query_features=self.params.get("query_features"),
            candidate_features=self.params.get("candidate_features"),
            account_gmv=None,
            chain_gmv=None,
            rating=None,
            monthly_orders=None,
            temperature=self.params.get("temperature"),
            unique_candidate=self.params.get("unique_candidate"),
            aggregation=self.params.get("aggregation"),
            num_hard_negatives=None,
            unique_candidates_features=None,
            batch_size=int(self.params.get("train_batch") * self.params.get("ns_ratio")),
            dropout=self.params.get("dropout"),
            enable_shared_chain_embedding=True,
            enable_shared_keywords_embedding=True,
            ese_vec_embedding_layer=ese_vec_embedding_layer,
            ese_ch_embedding_layer=ese_ch_embedding_layer,
            unique_geohash6_ids=params_input_data.unique_geohash6_ids,
            unique_area_ids=params_input_data.unique_area_ids,
            enable_shared_area_ids_embedding=True,
            unique_order_sources=self.params.get("unique_order_sources")
        )

        LOG.info("Building user model")
        user_embedding_model = self.model.customer_model
        sample_data_user = self.get_user_sample_input()
        user_embedding_model(sample_data_user)
        LOG.info("Loading weights into user model")
        user_embedding_model.load_weights(self.user_weights_file)
        self.user_model = user_embedding_model

        LOG.info("Building chain model")
        chain_embedding_model = self.model.chain_model
        sample_data_chain = self.get_chain_sample_input()
        chain_embedding_model(sample_data_chain)
        LOG.info("Loading weights into chain model")
        chain_embedding_model.load_weights(self.chain_weights_file)
        self.chain_model = chain_embedding_model

        self.tower_model_map = {
            'user_tower': self.user_model,
            'chain_tower': self.chain_model,
        }

        def _call(model, features):
            return model(features)

        self.model_tf_func = tf.function(_call)

        # WARM UP USER MODEL: BUILD TF GRAPH
        for model_input in (self.user_model_input, self.user_model_input_2, self.user_model_input_3):
            with perf_manager(
                    f"Warming up user model with signature: {sorted(list(model_input.keys()))}",
                    description_before="Warming up user model"
            ):
                self.get_embeddings('user_tower', model_input)

    @newrelic.agent.function_trace("inference:model_tf_func")
    def inference(self, model_tf, input_to_tf):
        embeddings = self.model_tf_func(model_tf, input_to_tf)
        return embeddings

    @newrelic.agent.function_trace()
    def get_embeddings(self, tower_name: str, model_input):
        # TODO: this handles only individual request not batch
        input_to_tf = self.input_to_tf_type(model_input)
        model_tf = self.tower_model_map[tower_name]

        embeddings = self.inference(model_tf, input_to_tf)
        return tf.squeeze(embeddings).numpy()

    def get_user_sample_input(self) -> dict:
        return self.input_to_tf_type(self.user_model_input)

    def get_chain_sample_input(self) -> dict:
        return self.input_to_tf_type(self.chain_model_input)

    @staticmethod
    def input_to_tf_type(model_input: dict):
        return {
            key: tf.constant(value, shape=(1,), dtype=DTYPE_MAPPING[type(value)])
            for key, value in model_input.items()
        }
