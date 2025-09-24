import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

from abstract_ranking.two_tower.transformer_encoder_layer import GlobalSelfAttention
from vendor_ranking.two_tower.user_models import create_vectorized_attention_layer, create_int_embedding_layer


class UserModelV1(tf.keras.Model):
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
                    # Encoder(d_model=chain_emb_size),
                    # tf.keras.layers.GlobalAveragePooling1D(),
                    search_agg,
                ],
                name="prev_searches",
            )
            self.prev_searches_vectorizer.adapt(searches_vocab)
            self.feat_to_partial_tower["prev_searches"] = self.prev_searches_embedding

        for feature in [
            "prev_clicks",
            "freq_clicks",
            "user_prev_chains",
            "freq_chains",
        ]:
            if feature in query_features:
                feature_tower = create_vectorized_attention_layer(
                    vocab=list(unique_chain_ids),
                    aggregation=aggregation,
                    embedding_size=chain_emb_size,
                    attention_dim=chain_emb_size,
                    layer_name_prefix=feature,
                    predefined_embedding=shared_chain_embedding,
                )
                setattr(self, feature + "_tower", feature_tower)

                self.feat_to_partial_tower[feature] = feature_tower

        ## defining ese layers
        if ese_vec_embedding_layer:
            for feature in [
                "user_prev_chains_ese",
                "freq_chains_ese",
                "freq_clicks_ese",
            ]:
                self.feat_to_partial_tower[feature] = ese_vec_embedding_layer

        delivery_area_feature_name = "delivery_area_id"
        if delivery_area_feature_name in query_features and len(unique_area_ids):
            embedding_layer = create_int_embedding_layer(
                layer_prefix=delivery_area_feature_name,
                vocab=unique_area_ids,
                embedding_size=16,
            )
            self.delivery_area_id_tower = embedding_layer
            self.feat_to_partial_tower[delivery_area_feature_name] = embedding_layer

        for feature_col, vocab_size in zip(["order_hour", "order_weekday"], [24, 7]):
            if feature_col in query_features:
                embedding_layer = create_int_embedding_layer(
                    layer_prefix=feature_col,
                    vocab=np.arange(vocab_size),
                    embedding_size=16,
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
                name="geohash_embedding_layer",
            )
            self.feat_to_partial_tower["geohash6"] = self.geohash_embedding

        if account_gmv is not None:
            self.user_gmv_model = tf.keras.layers.Normalization(
                axis=None, mean=account_gmv.mean(), variance=account_gmv.var()
            )
            self.feat_to_partial_tower["account_avg_gmv"] = lambda x: tf.reshape(self.user_gmv_model(x), (-1, 1))

    @tf.function
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
            "delivery_area_id",
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
            available_features = [feature for feature in self.feat_to_partial_tower.keys()]
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


class ChainModel(tf.keras.Model):
    def __init__(
        self,
        chain_emb_size,
        cuisine_emb_size,
        unique_chain_ids,
        unique_cuisines_names,
        candidates_features,
        chain_cuisine_feat,
        monthly_orders=None,
        rating=None,
        chain_gmv=None,
        shared_chain_embedding=None,
        searches_vocab=None,
        shared_search_embedding=None,
        ese_ch_embedding_layer=None,
    ):
        super().__init__()

        if shared_chain_embedding:
            embedding = shared_chain_embedding
        else:
            embedding = tf.keras.layers.Embedding(len(unique_chain_ids) + 1, chain_emb_size, name="chain_embedding")
        self.chain_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(vocabulary=unique_chain_ids, mask_token="0", name="chain_lookup"),
                embedding,
            ]
        )
        self.feat_to_partial_tower = {"chain_id": self.chain_embedding}

        if ese_ch_embedding_layer:
            self.feat_to_partial_tower["chain_ese_embedding"] = ese_ch_embedding_layer

        if chain_cuisine_feat:
            self.cuisine_vectorizer = tf.keras.layers.TextVectorization(
                vocabulary=list(unique_cuisines_names),
                standardize="lower_and_strip_punctuation",
            )

            self.cuisine_text_embedding = tf.keras.Sequential(
                [
                    self.cuisine_vectorizer,
                    tf.keras.layers.Embedding(len(unique_cuisines_names) + 2, cuisine_emb_size, mask_zero=True),
                    tf.keras.layers.GlobalAveragePooling1D(),
                ]
            )
            self.feat_to_partial_tower["tlabel"] = self.cuisine_text_embedding
        self.chain_cuisine_feat = chain_cuisine_feat
        if monthly_orders is not None:
            self.monthly_order = tf.keras.layers.Normalization(
                axis=None, mean=monthly_orders.mean(), variance=monthly_orders.var()
            )

            # self.feat_to_partial_tower['monthly_orders'] = self.monthly_order

            self.feat_to_partial_tower["monthly_orders"] = lambda x: tf.reshape(self.monthly_order(x), (-1, 1))

        if "chain_name" in candidates_features:
            self.chain_name_vectorizer = tf.keras.layers.TextVectorization(
                vocabulary=list(searches_vocab),
                standardize="lower_and_strip_punctuation",
            )
            if shared_search_embedding:
                chain_name_embedding = shared_search_embedding
            else:
                chain_name_embedding = tf.keras.layers.Embedding(len(unique_chain_ids) + 1, chain_emb_size)

            self.chain_name_embedding = tf.keras.Sequential(
                [
                    self.chain_name_vectorizer,
                    chain_name_embedding,
                    tf.keras.layers.GlobalAveragePooling1D(),
                ]
            )
            self.feat_to_partial_tower["chain_name"] = self.chain_name_embedding

        if rating is not None:
            self.rating = tf.keras.layers.Normalization(axis=None, mean=rating.mean(), variance=rating.var())

            self.feat_to_partial_tower["rating"] = lambda x: tf.reshape(self.rating(x), (-1, 1))
            # self.feat_to_partial_tower['rating'] = self.rating

        if chain_gmv is not None:
            self.chain_avg_gmv = tf.keras.layers.Normalization(
                axis=None, mean=chain_gmv.mean(), variance=chain_gmv.var()
            )
            self.feat_to_partial_tower["chain_avg_gmv"] = lambda x: tf.reshape(self.chain_avg_gmv(x), (-1, 1))
            # self.feat_to_partial_tower['rating'] = self.rating

    def call(self, chains_df):
        potential_features = [
            "chain_id",
            "tlabel",
            "monthly_orders",
            "rating",
            "chain_name",
            "chain_avg_gmv",
        ]
        feature_col_dict = {x: x for x in potential_features}
        feature_col_dict["chain_ese_embedding"] = "chain_id"

        if len(self.feat_to_partial_tower) > 1:
            available_features = [feature for feature in potential_features if feature in chains_df] + [
                "chain_ese_embedding"
            ]
            output = tf.concat(
                [
                    self.feat_to_partial_tower[feature](chains_df[feature_col_dict[feature]])
                    for feature in available_features
                ],
                axis=1,
            )
        elif self.chain_cuisine_feat:
            output = tf.concat(
                [
                    self.chain_embedding(chains_df["chain_id"]),
                    self.cuisine_text_embedding(chains_df["tlabel"]),
                ],
                axis=1,
            )
        else:
            output = self.chain_embedding(chains_df["chain_id"])
        return output


class AccountChainModel(tfrs.Model):
    def __init__(
        self,
        customer_model,
        chain_model,
        candidates_ds,
        query_features,
        candidate_features,
        handle_popularity_bias,
        label_smoothing=0.0,
        temperature=1,
        unique_candidate="chain_id",
        num_hard_negatives=None,
        unique_candidates_features=None,
        batch_size=0,
    ):
        super().__init__()
        self.chain_model: tf.keras.Model = chain_model
        self.customer_model: tf.keras.Model = customer_model

        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(candidates=candidates_ds.batch(128).map(self.chain_model)),
            remove_accidental_hits=True,
            temperature=temperature,
            num_hard_negatives=num_hard_negatives,
            # loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
        )
        self.query_features = query_features
        self.candidate_features = candidate_features
        self.handle_popularity_bias = handle_popularity_bias
        self.unique_candidate = unique_candidate
        self.unique_candidate_features = unique_candidates_features
        self.batch_size = batch_size

    def compute_loss(self, features, training=False):
        mns = features[1]
        features = features[0]
        customer_embeddings = self.customer_model({a: features[a] for a in self.query_features}, training=True)
        chain_embeddings = self.chain_model(
            {a: tf.concat([features[a], mns[a]], axis=0) for a in self.candidate_features},
            training=True,
        )
        # if 'candidate_probability' in features.columns:
        #    candidate_probability = features['candidate_probability']

        ## mixed negative sampling
        candidate_probability = tf.concat([features["candidate_probability"], mns["candidate_probability"]], axis=0)
        candidate_ids = tf.concat([features[self.unique_candidate], mns[self.unique_candidate]], axis=0)
        sample_weight = features["sample_weight"]

        if self.unique_candidate_features is not None:
            uniform_negatives = self.unique_candidate_features.sample(self.batch_size)
            negatives_embeddings = self.chain_model(uniform_negatives)
            chain_embeddings = tf.concat([chain_embeddings, negatives_embeddings], axis=0)
            negative_ids = uniform_negatives[self.unique_candidate]
            # sample_weight = uniform_negatives['sample_weight']
            candidate_probability = tf.concat(
                [
                    features["candidate_probability"],
                    tf.convert_to_tensor(uniform_negatives["candidate_probability"]),
                ],
                axis=0,
            )
            candidate_ids = tf.concat(
                [features[self.unique_candidate], tf.convert_to_tensor(negative_ids)],
                axis=0,
            )
            # sample_weight = tf.concat([features['sample_weight'],
            #                           tf.convert_to_tensor(sample_weight)], axis=0)

        if self.handle_popularity_bias:
            return self.task(
                customer_embeddings,
                chain_embeddings,
                compute_metrics=not training,
                candidate_sampling_probability=candidate_probability,
                candidate_ids=candidate_ids,
                sample_weight=sample_weight,
            )
        else:
            return self.task(
                customer_embeddings,
                chain_embeddings,
                compute_metrics=not training,
                candidate_ids=features["chain_id"],
            )


def create_two_tower_model(
    embedding_dimension,
    unique_customer_ids,
    unique_chain_ids,
    unique_cuisines_names,
    candidates_ds,
    items_vocab=None,
    searches_vocab=None,
    prev_chains_feat=False,
    enable_prev_searches=False,
    enable_prev_items=False,
    chain_cuisine_feat=False,
    handle_popularity_bias=False,
    query_features=["account_id"],
    candidate_features=["chain_id"],
    account_gmv=None,
    chain_gmv=None,
    rating=None,
    monthly_orders=None,
    temperature=1,
    label_smoothing=0,
    unique_candidate="chain_id",
    aggregation="AVG",
    num_hard_negatives=None,
    unique_candidates_features=None,
    batch_size=0,
    dropout=0,
    enable_shared_chain_embedding=False,
    enable_shared_keywords_embedding=False,
    ese_vec_embedding_layer=None,
    ese_ch_embedding_layer=None,
    unique_geohash6_ids=None,
    unique_area_ids=None,
    chain_features_df=None,
):
    if enable_shared_chain_embedding:
        shared_chain_embedding = tf.keras.layers.Embedding(
            len(unique_chain_ids) + 2,
            embedding_dimension,
            name="shared_embedding",
            mask_zero=True,
        )
    else:
        shared_chain_embedding = None
    if enable_shared_keywords_embedding:
        shared_search_embedding = tf.keras.layers.Embedding(
            len(searches_vocab) + 2,
            embedding_dimension,
            name="keyword_embedding",
            mask_zero=True,
        )
    else:
        shared_search_embedding = None

    normalization = tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=-1))
    customer_model = tf.keras.Sequential(
        [
            UserModelV1(
                user_emb_size=embedding_dimension,
                chain_emb_size=embedding_dimension,
                unique_account_ids=unique_customer_ids,
                unique_chain_ids=unique_chain_ids,
                prev_chains_feat=prev_chains_feat,
                enable_prev_searches=enable_prev_searches,
                searches_vocab=searches_vocab,
                enable_prev_items=enable_prev_items,
                items_vocab=items_vocab,
                account_gmv=account_gmv,
                query_features=query_features,
                aggregation=aggregation,
                shared_chain_embedding=shared_chain_embedding,
                shared_search_embedding=shared_search_embedding,
                ese_vec_embedding_layer=ese_vec_embedding_layer,
                unique_geohash6_ids=unique_geohash6_ids,
                unique_area_ids=unique_area_ids,
            ),
            tf.keras.layers.BatchNormalization(),
            tfrs.layers.dcn.Cross(kernel_initializer="glorot_uniform"),
            tf.keras.layers.Dense(embedding_dimension * 4, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(embedding_dimension * 2, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32),
            normalization,
        ],
        name="query_tower",
    )
    chain_model = tf.keras.Sequential(
        [
            ChainModel(
                chain_emb_size=embedding_dimension,
                cuisine_emb_size=embedding_dimension,
                unique_chain_ids=unique_chain_ids,
                unique_cuisines_names=unique_cuisines_names,
                chain_cuisine_feat=chain_cuisine_feat,
                rating=rating,
                monthly_orders=monthly_orders,
                candidates_features=candidate_features,
                chain_gmv=chain_gmv,
                shared_chain_embedding=shared_chain_embedding,
                searches_vocab=searches_vocab,
                shared_search_embedding=shared_search_embedding,
                ese_ch_embedding_layer=ese_ch_embedding_layer,
            ),
            tf.keras.layers.BatchNormalization(),
            tfrs.layers.dcn.Cross(kernel_initializer="glorot_uniform"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(embedding_dimension * 2, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32),
            normalization,
        ],
        name="candidate_tower",
    )
    model = AccountChainModel(
        customer_model,
        chain_model,
        candidates_ds=candidates_ds,
        query_features=query_features,
        candidate_features=candidate_features,
        handle_popularity_bias=handle_popularity_bias,
        label_smoothing=label_smoothing,
        temperature=temperature,
        unique_candidate=unique_candidate,
        num_hard_negatives=num_hard_negatives,
        unique_candidates_features=unique_candidates_features,
        batch_size=batch_size,
        ##chain_features_df=chain_features_df
    )
    model.compile(optimizer=tf.keras.optimizers.Adam())

    return model


def create_user_model_from_config(**params):
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
            ),
            # tf.keras.layers.Dense(embedding_dimension, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(params["dropout"]),
            tf.keras.layers.Dense(params["embedding_dimension"] * 2, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(params["dropout"] / 2),
            tf.keras.layers.Dense(32),
            # tf.keras.layers.LayerNormalization(axis=-1)
            normalization,
        ],
        name="query_tower",
    )

    return customer_model


def create_v1_user_model_from_config(**params):
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
            ),
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
