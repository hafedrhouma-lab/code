import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

from vendor_ranking.two_tower.ese_embedding_layer import get_ese_embedding_tf_layer
from abstract_ranking.two_tower.transformer_encoder_layer import GlobalSelfAttention
from vendor_ranking.two_tower.user_models import (
    create_string_embedding_layer,
    create_int_embedding_layer,
    create_vectorized_attention_layer
)
from vendor_ranking.two_tower.user_models.v2 import UserModelV2


def create_v3_user_model_from_config(**params):
    if params["enable_shared_chain_embedding"]:
        shared_chain_embedding = tf.keras.layers.Embedding(
            len(params["unique_chain_ids"]) + 2,
            params["embedding_dimension"],
            name="shared_embedding",
            mask_zero=True,
        )
    else:
        shared_chain_embedding = None

    if params["enable_shared_keywords_embedding"]:
        shared_search_embedding = tf.keras.layers.Embedding(
            len(params["searches_vocab"]) + 2,
            params["embedding_dimension"],
            name="keyword_embedding",
            mask_zero=True,
        )
    else:
        shared_search_embedding = None

    normalization = tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=-1))

    ese_vec_embedding_layer = get_ese_embedding_tf_layer(
        embedding_date="2023-07-01", country_code=params["country"], is_vector=True
    )

    customer_model = tf.keras.Sequential(
        [
            UserModelV2(
                user_emb_size=params["embedding_dimension"],
                chain_emb_size=params["embedding_dimension"],
                unique_account_ids=params["unique_customer_ids"],
                unique_chain_ids=params["unique_chain_ids"],
                prev_chains_feat=params["prev_chains_feat"],
                enable_prev_searches=params["enable_prev_searches"],
                searches_vocab=params["searches_vocab"],
                enable_prev_items=params["enable_prev_items"],
                items_vocab=params["items_vocab"],
                account_gmv=params["account_gmv"],
                query_features=params["query_features"],
                aggregation=params["aggregation"],
                shared_chain_embedding=shared_chain_embedding,
                shared_search_embedding=shared_search_embedding,
                ese_vec_embedding_layer=ese_vec_embedding_layer,
                unique_geohash6_ids=params["unique_geohash6_ids"],
            ),
            tf.keras.layers.BatchNormalization(),
            tfrs.layers.dcn.Cross(kernel_initializer="glorot_uniform"),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(params["embedding_dimension"] * 4, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(params["dropout"] / 2),
            tf.keras.layers.Dense(params["embedding_dimension"] * 2, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(params["dropout"] / 2),
            tf.keras.layers.Dense(32),
            normalization,
        ],
        name="query_tower",
    )

    return customer_model


class UserModelV3(tf.keras.Model):
    def __init__(
        self,
        user_emb_size,
        chain_emb_size,
        unique_account_ids,
        unique_chain_ids,
        prev_chains_feat,
        enable_prev_searches=False,
        searches_vocab=None,
        enable_prev_items=False,
        items_vocab=None,
        account_gmv=None,
        query_features=None,
        aggregation="AVG",
        shared_chain_embedding=None,
        shared_search_embedding=None,
        ese_vec_embedding_layer=None,
        unique_geohash6_ids=None,
        unique_area_ids=None,
        shared_area_embedding=None,
        unique_order_sources=None
    ):
        super().__init__()
        self.prev_chains_feat = prev_chains_feat
        self.feat_to_partial_tower = {}
        if "account_id" in query_features:
            self.account_embedding = tf.keras.Sequential(
                [
                    tf.keras.layers.StringLookup(
                        vocabulary=unique_account_ids,
                        mask_token=None,
                        name="account_lookup",
                    ),
                    tf.keras.layers.Embedding(
                        len(unique_account_ids) + 1,
                        user_emb_size,
                        name="account_embedding",
                    ),
                ]
            )
            self.feat_to_partial_tower["account_id"] = self.account_embedding

        if enable_prev_items:
            max_tokens = len(items_vocab)
            self.prev_items_vectorizer = tf.keras.layers.TextVectorization(
                max_tokens=max_tokens,
                standardize="strip_punctuation",
                output_sequence_length=30,
            )
            if aggregation == "AVG":
                items_agg = tf.keras.layers.GlobalAveragePooling1D()
            else:
                items_agg = tf.keras.layers.GRU(units=chain_emb_size)
            self.prev_items_embedding = tf.keras.Sequential(
                [
                    self.prev_items_vectorizer,
                    tf.keras.layers.Embedding(
                        max_tokens, chain_emb_size, mask_zero=True
                    ),
                    GlobalSelfAttention(
                        num_heads=1, key_dim=chain_emb_size, dropout=0.1
                    ),
                    items_agg,
                ]
            )
            self.prev_items_vectorizer.adapt(items_vocab)
            self.feat_to_partial_tower["prev_items"] = self.prev_items_embedding

        if enable_prev_searches:
            if aggregation == "AVG":
                search_agg = tf.keras.layers.GlobalAveragePooling1D()
            else:
                search_agg = tf.keras.layers.GRU(units=chain_emb_size)
            self.prev_searches_vectorizer = tf.keras.layers.TextVectorization(
                vocabulary=searches_vocab,
                standardize="strip_punctuation",
                output_sequence_length=15,
            )
            if shared_search_embedding:
                embedding = shared_search_embedding
            else:
                embedding = tf.keras.layers.Embedding(
                    len(searches_vocab) + 2, chain_emb_size, mask_zero=True
                )

            self.prev_searches_embedding = tf.keras.Sequential(
                [
                    self.prev_searches_vectorizer,
                    embedding,
                    GlobalSelfAttention(
                        num_heads=1, key_dim=chain_emb_size, dropout=0.1
                    ),
                    search_agg,
                ],
                name="prev_searches"
            )
            self.feat_to_partial_tower["prev_searches"] = self.prev_searches_embedding

        for feature in ["prev_clicks", "freq_clicks", "user_prev_chains", "freq_chains"]:
            if feature in query_features:
                feature_tower = create_vectorized_attention_layer(
                    vocab=list(unique_chain_ids), aggregation=aggregation,
                    embedding_size=chain_emb_size, attention_dim=chain_emb_size,
                    layer_name_prefix=feature, predefined_embedding=shared_chain_embedding
                )
                setattr(self, feature + "_tower", feature_tower)

                self.feat_to_partial_tower[feature] = feature_tower

        ## defining ese layers
        if ese_vec_embedding_layer:
            for feature in ["user_prev_chains_ese", "freq_chains_ese", "freq_clicks_ese"]:
                self.feat_to_partial_tower[feature] = ese_vec_embedding_layer

        # define order source embedding
        if unique_order_sources is not None and len(unique_order_sources) > 1 \
            and "account_order_source" in query_features:
            self.order_source_tower = create_string_embedding_layer(layer_prefix="order_source",
                                                                    vocab=unique_order_sources,
                                                                    embedding_size=chain_emb_size)

            self.feat_to_partial_tower["account_order_source"] = self.order_source_tower

        delivery_area_feature_name = "delivery_area_id"
        if delivery_area_feature_name in query_features and unique_area_ids is not None:

            if shared_area_embedding:
                area_embedding = shared_area_embedding
            else:
                area_embedding = tf.keras.layers.Embedding(
                    len(unique_area_ids) + 1, chain_emb_size // 2, name="area_embedding"
                )
            self.area_embedding = tf.keras.Sequential(
                [
                    tf.keras.layers.StringLookup(
                        vocabulary=unique_area_ids, mask_token="0", name="area_lookup"
                    ),
                    area_embedding,
                ]
            )
            self.delivery_area_id_tower = self.area_embedding
            self.feat_to_partial_tower[delivery_area_feature_name] = self.area_embedding

        for feature_col, vocab_size in zip(["order_hour", "order_weekday"], [24, 7]):

            if feature_col in query_features:
                embedding_layer = create_int_embedding_layer(
                    layer_prefix=feature_col,
                    vocab=np.arange(vocab_size),
                    embedding_size=16
                )
                setattr(self, feature_col + "_tower", embedding_layer)
                self.feat_to_partial_tower[feature_col] = embedding_layer

        if unique_geohash6_ids is not None and len(unique_geohash6_ids):
            self.geohash_embedding = tf.keras.Sequential(
                [
                    tf.keras.layers.StringLookup(
                        vocabulary=unique_geohash6_ids,
                        mask_token=None,
                        name="geohash6_lookup",
                    ),
                    tf.keras.layers.Embedding(
                        len(unique_geohash6_ids) + 1,
                        chain_emb_size // 2,
                        name="geohash6_embedding",
                    ),
                ],
                name="geohash_embedding_layer"
            )
            self.feat_to_partial_tower["geohash6"] = self.geohash_embedding

        if account_gmv is not None:
            self.user_gmv_model = tf.keras.layers.Normalization(
                axis=None, mean=account_gmv.mean(), variance=account_gmv.var()
            )
            self.feat_to_partial_tower["account_avg_gmv"] = lambda x: tf.reshape(
                self.user_gmv_model(x), (-1, 1)
            )

        self.numeric_features = []
        self.query_features = query_features
        for chain_feature in self.query_features:
            num_flags = ["log", "avg", "pct", "eur", "cnt", "tpro"]
            for flag in num_flags:
                if flag in chain_feature:
                    self.numeric_features.append(chain_feature)
                    break
        self.numeric_features = sorted(self.numeric_features)
        num_features_length = len(self.numeric_features)
        if num_features_length > 0:
            for numeric_feature in self.numeric_features:
                numeric_features_input_layer = tf.keras.layers.Input(
                    shape=(1), name=numeric_feature + "_input"
                )
                numeric_feature_input = tf.keras.layers.Lambda(lambda x: tf.reshape(
                    tf.cast(x, tf.float32), (-1, 1)
                ))
                self.feat_to_partial_tower[numeric_feature] = tf.keras.Sequential(
                    [numeric_features_input_layer, numeric_feature_input])

    @tf.function
    def call(self, features_df):
        potential_features = self.numeric_features + [
            "user_prev_chains",
            "prev_items",
            "prev_searches",
            "prev_clicks",
            "account_id",
            "order_hour",
            "order_weekday",
            "account_avg_gmv",
            "freq_clicks",
            "freq_chains",
            "freq_clicks_ese",
            "freq_chains_ese",
            "user_prev_chains_ese",
            "geohash6",
            "delivery_area_id",
            "account_order_source"
        ]
        feature_col_dict = {x: x for x in potential_features}
        feature_col_dict["freq_clicks_ese"] = "freq_clicks"
        feature_col_dict["freq_chains_ese"] = "freq_chains"
        feature_col_dict["user_prev_chains_ese"] = "user_prev_chains"
        if len(self.feat_to_partial_tower) > 1:
            available_features = [
                                     feature for feature in potential_features if feature in features_df
                                 ] + ["freq_clicks_ese", "freq_chains_ese", "user_prev_chains_ese"]
            available_features = [feature for feature in self.feat_to_partial_tower.keys()]
            output = tf.concat(
                [
                    self.feat_to_partial_tower[feature](features_df[feature_col_dict[feature]])
                    for feature in available_features if feature in self.feat_to_partial_tower
                ],
                axis=1,
            )
        elif self.prev_chains_feat:
            output = tf.concat(
                [
                    self.account_embedding(features_df["account_id"]),
                    self.prev_chains_embedding(features_df["user_prev_chains"]),
                    self.hour_embedding(features_df["order_hour"]),
                    self.weekday_embedding(features_df["order_weekday"]),
                ],
                axis=1,
            )
        else:
            output = self.account_embedding(features_df["account_id"])

        return output


def create_tt_v3_user_model_from_config(**params):
    ese_vec_embedding_layer = get_ese_embedding_tf_layer(embedding_date="2023-07-01",
                                                         country_code=params["country"],
                                                         is_vector=True,
                                                         embeddings_df=params["ese_embeddings_df"])
    if params["enable_shared_chain_embedding"]:
        shared_chain_embedding = tf.keras.layers.Embedding(len(params["unique_chain_ids"]) + 2,
                                                           params["embedding_dimension"],
                                                           name="shared_embedding", mask_zero=True)
    else:
        shared_chain_embedding = None
    if params["enable_shared_keywords_embedding"]:
        shared_search_embedding = tf.keras.layers.Embedding(len(params["searches_vocab"]) + 2,
                                                            params["embedding_dimension"],
                                                            name="keyword_embedding", mask_zero=True)
    else:
        shared_search_embedding = None

    use_shared_area_embedding = type(params["unique_area_ids"][0]) is str
    if use_shared_area_embedding:
        shared_area_embedding = tf.keras.layers.Embedding(
            len(params["unique_area_ids"]) + 2,
            params["embedding_dimension"],
            name="area_id_embedding",
            mask_zero=True,
        )
    else:
        shared_area_embedding = None

    normalization = tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=-1))
    customer_model = tf.keras.Sequential([
        UserModelV3(user_emb_size=params["embedding_dimension"], chain_emb_size=params["embedding_dimension"],
                    unique_account_ids=params["unique_customer_ids"], unique_chain_ids=params["unique_chain_ids"],
                    prev_chains_feat=params["prev_chains_feat"],
                    enable_prev_searches=params["enable_prev_searches"], searches_vocab=params["searches_vocab"],
                    enable_prev_items=params["enable_prev_items"], items_vocab=params["items_vocab"],
                    account_gmv=params["account_gmv"], query_features=params["query_features"],
                    aggregation=params["aggregation"], shared_chain_embedding=shared_chain_embedding,
                    shared_search_embedding=shared_search_embedding,
                    ese_vec_embedding_layer=ese_vec_embedding_layer,
                    unique_geohash6_ids=params["unique_geohash6_ids"],
                    unique_area_ids=params["unique_area_ids"],
                    shared_area_embedding=shared_area_embedding,
                    unique_order_sources=params["unique_order_sources"]
                    ),
        tf.keras.layers.BatchNormalization(),
        tfrs.layers.dcn.Cross(kernel_initializer="glorot_uniform"),
        tf.keras.layers.Dense(params["embedding_dimension"] * 4, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(params["embedding_dimension"] * 2, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32),
        normalization,
    ], name="query_tower")

    return customer_model
