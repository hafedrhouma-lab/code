import logging
import mlflow
import tensorflow as tf

from base.v0.perf import perf_manager

from projects.vendor_ranking.common.two_towers.src.cli.prepare.feast.prepare_param import PrepareData
from projects.vendor_ranking.common.two_towers.src.cli.prepare.feast.prepare_utils import PrepareUtils

from ..data.datasets.tf_data import (
    create_tf_dataset_from_tensor_slices
)

from ..model.ese_embedding_layer import (
    get_ese_embedding_tf_layer,
)
from ..model.recall_at_k_callback import OrderLevelRecallatK
from ..model.two_tower_model import create_two_tower_model

logger = tf.get_logger()
logger.setLevel(logging.ERROR)


def train_model(
        train_ds: tf.data.Dataset,
        params: dict,
        params_input_data: PrepareData,
        data_training_utils: PrepareUtils
):
    ese_vec_embedding_layer = get_ese_embedding_tf_layer(
        is_vector=True,
        embeddings_df=data_training_utils.ese_chain_embeddings
    )
    ese_ch_embedding_layer = get_ese_embedding_tf_layer(
        is_vector=False,
        embeddings_df=data_training_utils.ese_chain_embeddings
    )

    candidates_ds = create_tf_dataset_from_tensor_slices(
        params_input_data.chain_features_df
    )

    recall_callback = OrderLevelRecallatK(
        k=[10, 20, 30, 50, 200],
        geo_to_chains=None,  # data_training_utils.geo_to_chains,
        geo_to_parent_geo=None,  # data_training_utils.geo_to_parent_geo,
        test_df=params_input_data.test_df,
        train_sample_df=None,
        query_features=params.get("query_features"),
        candidate_features=params.get("candidate_features"),
        account_prev_interactions=None,
        chain_features_df=params_input_data.chain_features_df,
        chain_names=params_input_data.chain_names,
        logger=logger,
    )

    callbacks = [recall_callback]

    model = create_two_tower_model(
        embedding_dimension=params.get("embedding_dimension"),
        unique_customer_ids=params.get("unique_customer_ids"),
        unique_chain_ids=params_input_data.unique_chain_ids,
        unique_cuisines_names=params_input_data.unique_cuisines_names,
        candidates_ds=candidates_ds,
        prev_chains_feat=True,
        enable_prev_searches=params.get("enable_prev_searches"),
        enable_prev_items=params.get("enable_prev_items"),
        items_vocab=params.get("items_vocab_list"),
        searches_vocab=params_input_data.search_vocab_list,
        chain_cuisine_feat=True,
        handle_popularity_bias=True,
        query_features=params.get("query_features"),
        candidate_features=params.get("candidate_features"),
        account_gmv=None,
        chain_gmv=None,
        rating=None,
        monthly_orders=None,
        temperature=params.get("temperature"),
        unique_candidate=params.get("unique_candidate"),
        aggregation=params.get("aggregation"),
        num_hard_negatives=None,
        unique_candidates_features=None,
        batch_size=int(params.get("train_batch") * params.get("ns_ratio")),
        dropout=params.get("dropout"),
        enable_shared_chain_embedding=True,
        enable_shared_keywords_embedding=True,
        ese_vec_embedding_layer=ese_vec_embedding_layer,
        ese_ch_embedding_layer=ese_ch_embedding_layer,
        unique_geohash6_ids=params_input_data.unique_geohash6_ids,
        unique_area_ids=params_input_data.unique_area_ids,
        enable_shared_area_ids_embedding=True,
        unique_order_sources=params.get("unique_order_sources")
    )

    epochs = params.get("num_epochs")
    with perf_manager(
            description="Finished Model Training",
            description_before=f"Training Model {epochs=}..."
    ):
        model.fit(
            train_ds,
            epochs=epochs,
            callbacks=callbacks
        )

    if mlflow.active_run():
        for epoch, loss_value in enumerate(model.history.history['loss'], start=1):
            mlflow.log_metric(f"training_loss", loss_value, step=epoch)

    recall_values = recall_callback.recall_list

    return model, recall_values
