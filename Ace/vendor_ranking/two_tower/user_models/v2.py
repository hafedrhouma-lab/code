import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

from vendor_ranking.two_tower.ese_embedding_layer import get_ese_embedding_tf_layer
from abstract_ranking.two_tower.transformer_encoder_layer import GlobalSelfAttention
from vendor_ranking.two_tower.user_models.v1 import UserModelV1, ChainModel


class UserModelV2(tf.keras.Model):
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
        if prev_chains_feat:
            self.chain_vectorizer = tf.keras.layers.TextVectorization(
                vocabulary=list(unique_chain_ids), standardize="strip_punctuation"
            )
            if aggregation == "AVG":
                chains_agg = tf.keras.layers.GlobalAveragePooling1D()
            else:
                chains_agg = tf.keras.layers.GRU(units=chain_emb_size)
            if shared_chain_embedding:
                embedding = shared_chain_embedding
            else:
                embedding = tf.keras.layers.Embedding(len(unique_chain_ids) + 2, chain_emb_size, mask_zero=True)
            self.prev_chains_embedding = tf.keras.Sequential(
                [
                    self.chain_vectorizer,
                    embedding,
                    GlobalSelfAttention(num_heads=1, key_dim=chain_emb_size, dropout=0.1),
                    chains_agg,
                ]
            )
            self.feat_to_partial_tower["user_prev_chains"] = self.prev_chains_embedding
            if ese_vec_embedding_layer:
                self.feat_to_partial_tower["user_prev_chains_ese"] = ese_vec_embedding_layer
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
                    tf.keras.layers.Embedding(max_tokens, chain_emb_size, mask_zero=True),
                    GlobalSelfAttention(num_heads=1, key_dim=chain_emb_size, dropout=0.1),
                    # tf.keras.layers.GlobalAveragePooling1D(),
                    items_agg,
                ]
            )
            self.prev_items_vectorizer.adapt(items_vocab)
            self.feat_to_partial_tower["prev_items"] = self.prev_items_embedding

        if enable_prev_searches:
            max_tokens = len(searches_vocab)
            if aggregation == "AVG":
                search_agg = tf.keras.layers.GlobalAveragePooling1D()
            else:
                search_agg = tf.keras.layers.GRU(units=chain_emb_size)
            self.prev_searches_vectorizer = tf.keras.layers.TextVectorization(
                max_tokens=max_tokens,
                standardize="strip_punctuation",
                output_sequence_length=15,
            )
            if shared_search_embedding:
                embedding = shared_search_embedding
            else:
                embedding = tf.keras.layers.Embedding(len(searches_vocab) + 2, chain_emb_size, mask_zero=True)

            self.prev_searches_embedding = tf.keras.Sequential(
                [
                    self.prev_searches_vectorizer,
                    embedding,
                    GlobalSelfAttention(num_heads=1, key_dim=chain_emb_size, dropout=0.1),
                    # EncoderLayer(d_model=chain_emb_size//2, num_heads=1, dff=chain_emb_size),
                    # tf.keras.layers.GlobalAveragePooling1D(),
                    search_agg,
                ],
                name="prev_searches",
            )
            self.prev_searches_vectorizer.adapt(searches_vocab)
            self.feat_to_partial_tower["prev_searches"] = self.prev_searches_embedding

        if "prev_clicks" in query_features:
            self.clicks_vectorizer = tf.keras.layers.TextVectorization(
                vocabulary=list(unique_chain_ids), standardize="strip_punctuation"
            )
            if aggregation == "AVG":
                clicks_agg = tf.keras.layers.GlobalAveragePooling1D()
            else:
                clicks_agg = tf.keras.layers.GRU(units=chain_emb_size)
            if shared_chain_embedding:
                embedding = shared_chain_embedding
            else:
                embedding = tf.keras.layers.Embedding(len(unique_chain_ids) + 2, chain_emb_size, mask_zero=True)
            self.prev_clicks_embedding = tf.keras.Sequential(
                [
                    self.chain_vectorizer,
                    embedding,
                    GlobalSelfAttention(num_heads=1, key_dim=chain_emb_size, dropout=0.1),
                    clicks_agg,
                ],
                name="prev_clicks_layer",
            )
            self.feat_to_partial_tower["prev_clicks"] = self.prev_clicks_embedding

        if "freq_clicks" in query_features:
            self.freq_clicks_vectorizer = tf.keras.layers.TextVectorization(
                vocabulary=list(unique_chain_ids), standardize="strip_punctuation"
            )
            if aggregation == "AVG":
                freq_clicks_agg = tf.keras.layers.GlobalAveragePooling1D()
            else:
                freq_clicks_agg = tf.keras.layers.GRU(units=chain_emb_size)
            if shared_chain_embedding:
                freqcl_embedding = shared_chain_embedding
            else:
                freqcl_embedding = tf.keras.layers.Embedding(len(unique_chain_ids) + 2, chain_emb_size, mask_zero=True)
            self.freq_clicks_embedding = tf.keras.Sequential(
                [
                    self.freq_clicks_vectorizer,
                    freqcl_embedding,
                    GlobalSelfAttention(num_heads=1, key_dim=chain_emb_size, dropout=0.1),
                    freq_clicks_agg,
                ],
                name="freq_clicks_layer",
            )
            self.feat_to_partial_tower["freq_clicks"] = self.freq_clicks_embedding
            if ese_vec_embedding_layer:
                self.feat_to_partial_tower["freq_clicks_ese"] = ese_vec_embedding_layer

        if "freq_chains" in query_features:
            self.freq_chains_vectorizer = tf.keras.layers.TextVectorization(
                vocabulary=list(unique_chain_ids), standardize="strip_punctuation"
            )
            if aggregation == "AVG":
                freq_chains_agg = tf.keras.layers.GlobalAveragePooling1D()
            else:
                freq_chains_agg = tf.keras.layers.GRU(units=chain_emb_size)
            if shared_chain_embedding:
                freqch_embedding = shared_chain_embedding
            else:
                freqch_embedding = tf.keras.layers.Embedding(len(unique_chain_ids) + 2, chain_emb_size, mask_zero=True)
            self.freq_chains_embedding = tf.keras.Sequential(
                [
                    self.freq_chains_vectorizer,
                    freqch_embedding,
                    GlobalSelfAttention(num_heads=1, key_dim=chain_emb_size, dropout=0.1),
                    freq_chains_agg,
                ],
                name="freq_chains_layer",
            )
            self.feat_to_partial_tower["freq_chains"] = self.freq_chains_embedding
            if ese_vec_embedding_layer:
                self.feat_to_partial_tower["freq_chains_ese"] = ese_vec_embedding_layer

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
                        chain_emb_size,
                        name="geohash6_embedding",
                    ),
                ],
                name="geohash_embedding_layer",
            )
            self.feat_to_partial_tower["geohash6"] = self.geohash_embedding

        # will be connected to the model only if features provided
        if "order_hour" in query_features:
            self.hour_embedding = tf.keras.Sequential(
                [
                    tf.keras.layers.IntegerLookup(vocabulary=np.arange(24), mask_token=None, name="order_hour"),
                    tf.keras.layers.Embedding(25, 16),
                ]
            )
            self.feat_to_partial_tower["order_hour"] = self.hour_embedding

        if "order_weekday" in query_features:
            self.weekday_embedding = tf.keras.Sequential(
                [
                    tf.keras.layers.IntegerLookup(vocabulary=np.arange(7), mask_token=None, name="order_weekday"),
                    tf.keras.layers.Embedding(8, 16),
                ],
                name="order_weekday_layer",
            )
            self.feat_to_partial_tower["order_weekday"] = self.weekday_embedding
        if account_gmv is not None:
            self.user_gmv_model = tf.keras.layers.Normalization(
                axis=None, mean=account_gmv.mean(), variance=account_gmv.var()
            )
            self.feat_to_partial_tower["account_avg_gmv"] = lambda x: tf.reshape(self.user_gmv_model(x), (-1, 1))

    def call(self, features_df):
        potential_features = [
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
        ]
        feature_col_dict = {x: x for x in potential_features}
        feature_col_dict["freq_clicks_ese"] = "freq_clicks"
        feature_col_dict["freq_chains_ese"] = "freq_chains"
        feature_col_dict["user_prev_chains_ese"] = "user_prev_chains"

        if len(self.feat_to_partial_tower) > 1:
            available_features = [feature for feature in potential_features if feature in features_df] + [
                "freq_clicks_ese",
                "freq_chains_ese",
                "user_prev_chains_ese",
            ]
            output = tf.concat(
                [
                    self.feat_to_partial_tower[feature](features_df[feature_col_dict[feature]])
                    for feature in available_features
                    if feature in self.feat_to_partial_tower
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


def create_tt_v2_user_model_from_config(**params):
    ese_vec_embedding_layer = get_ese_embedding_tf_layer(
        embedding_date="2023-07-01",
        country_code=params["country"],
        is_vector=True,
        embeddings_df=params["ese_embeddings_df"],
    )
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
    customer_model = tf.keras.Sequential(
        [
            UserModelV1(
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
                unique_area_ids=params["unique_area_ids"],
            ),
            # tf.keras.layers.BatchNormalization(),
            # tfrs.layers.dcn.Cross(),
            # tf.keras.layers.Dropout(params['dropout']),
            # tf.keras.layers.Dense(params['embedding_dimension'] * 2, activation="relu"),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Dropout(params['dropout']),
            # tf.keras.layers.Dense(32),
            # normalization,
            tf.keras.layers.BatchNormalization(),
            tfrs.layers.dcn.Cross(kernel_initializer="glorot_uniform"),
            tf.keras.layers.Dense(params["embedding_dimension"] * 4, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(params["embedding_dimension"] * 2, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32),
            normalization,
        ],
        name="query_tower",
    )

    return customer_model


def create_tt_v2_chain_model_from_config(**params):
    ese_ch_embedding_layer = get_ese_embedding_tf_layer(
        embedding_date="2023-07-01",
        country_code=params["country"],
        is_vector=False,
        embeddings_df=params["ese_embeddings_df"],
    )
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

    chain_model = tf.keras.Sequential(
        [
            ChainModel(
                chain_emb_size=params["embedding_dimension"],
                cuisine_emb_size=params["embedding_dimension"],
                unique_chain_ids=params["unique_chain_ids"],
                unique_cuisines_names=params["unique_cuisines_names"],
                chain_cuisine_feat=params["chain_cuisine_feat"],
                rating=params["rating"],
                monthly_orders=params["monthly_orders"],
                candidates_features=params["candidate_features"],
                chain_gmv=params["chain_gmv"],
                shared_chain_embedding=shared_chain_embedding,
                searches_vocab=params["searches_vocab"],
                shared_search_embedding=shared_search_embedding,
                ese_ch_embedding_layer=ese_ch_embedding_layer,
            ),
            tf.keras.layers.BatchNormalization(),
            tfrs.layers.dcn.Cross(kernel_initializer="glorot_uniform"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(params["embedding_dimension"] * 2, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32),
            normalization,
        ],
        name="candidate_tower",
    )

    return chain_model


def create_v2_user_model_from_config(**params):
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
            tfrs.layers.dcn.Cross(kernel_initializer="glorot_uniform"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(params["dropout"]),
            tf.keras.layers.Dense(params["embedding_dimension"] * 2, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(params["dropout"] / 2),
            tf.keras.layers.Dense(32),
            normalization,
        ],
        name="query_tower",
    )

    return customer_model
