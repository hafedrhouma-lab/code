import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

from abstract_ranking.two_tower.transformer_encoder_layer import GlobalSelfAttention


def create_vectorized_attention_layer(vocab, aggregation,
                                      emebdding_size,
                                      attention_dim,
                                      layer_name_prefix,
                                      predefined_embedding=None,
                                      dropout=0.1,
                                      apply_attention=True
                                      ):
    vectorizer = tf.keras.layers.TextVectorization(
        vocabulary=list(vocab), standardize="lower_and_strip_punctuation",
        name=layer_name_prefix + '_vectorizer',
        output_sequence_length=30
    )
    if aggregation == "AVG":
        chains_agg = tf.keras.layers.GlobalAveragePooling1D(name=layer_name_prefix + '_avg_pooling')
    else:
        chains_agg = tf.keras.layers.GRU(units=emebdding_size)
    if predefined_embedding:
        embedding = predefined_embedding
    else:
        embedding = tf.keras.layers.Embedding(
            len(vocab) + 2, emebdding_size, mask_zero=True,
            name=layer_name_prefix + '_embedding'
        )
    if apply_attention:
        attention_layer = GlobalSelfAttention(
            num_heads=1, key_dim=attention_dim, dropout=dropout,
            name=layer_name_prefix + '_attention'
        )
        layers = [vectorizer, embedding, attention_layer, chains_agg]
    else:
        layers = [vectorizer, embedding, chains_agg]
    vectorized_embedding = tf.keras.Sequential(
        layers
        ,
        name=layer_name_prefix + '_tower'
    )
    return vectorized_embedding


def create_int_embedding_layer(layer_prefix, vocab, embedding_size):
    embedding_layer = tf.keras.Sequential(
        [
            tf.keras.layers.IntegerLookup(
                vocabulary=vocab, mask_token=None, name=layer_prefix + "_lookup"
            ),
            tf.keras.layers.Embedding(len(vocab) + 1, embedding_size, name=layer_prefix + '_embedding'),
        ],
        name=layer_prefix + '_layer'
    )
    return embedding_layer


def create_string_embedding_layer(layer_prefix, vocab, embedding_size):
    embedding_layer = tf.keras.Sequential(
        [
            tf.keras.layers.StringLookup(
                vocabulary=vocab, mask_token=None, name=layer_prefix + "_lookup"
            ),
            tf.keras.layers.Embedding(len(vocab) + 1, embedding_size, name=layer_prefix + '_embedding'),
        ],
        name=layer_prefix + '_layer'
    )
    return embedding_layer


def create_numeric_input_layer(layer_prefix):
    numeric_layer = tf.keras.layers.Input(axis=None, name=layer_prefix + '_layer')
    return numeric_layer


class userModel_v1(tf.keras.Model):
    def __init__(
            self,
            item_emb_size,
            unique_chain_ids,
            unique_item_ids,
            unique_items_vocab,
            unique_area_ids=None,
            query_features=None,
            aggregation="AVG",
            shared_item_embeddings=None,
            shared_item_vocab_embedding=None,
    ):
        super().__init__()
        self.feat_to_partial_tower = {}

        for feature in ['prev_items', 'freq_items', 'prev_items_names', 'freq_items_names',
                        'chain_prev_items', 'chain_prev_items_names']:
            vocab = unique_item_ids if 'names' not in feature else unique_items_vocab
            shared_embedding = shared_item_embeddings if 'names' not in feature else shared_item_vocab_embedding
            if feature in query_features:
                feature_tower = create_vectorized_attention_layer(
                    vocab=list(vocab), aggregation=aggregation,
                    emebdding_size=item_emb_size, attention_dim=item_emb_size,
                    layer_name_prefix=feature, predefined_embedding=shared_embedding
                )
                setattr(self, feature + '_tower', feature_tower)

                self.feat_to_partial_tower[feature] = feature_tower

        if 'chain_id' in query_features:
            embedding = tf.keras.layers.Embedding(
                len(unique_chain_ids) + 1, item_emb_size, name="chain_embedding"
            )
            self.chain_embedding = tf.keras.Sequential(
                [
                    tf.keras.layers.StringLookup(
                        vocabulary=unique_chain_ids, mask_token=None, name="chain_lookup"
                    ),
                    embedding,
                ],
                name='chain_id' + '_layer'
            )
            self.feat_to_partial_tower["chain_id"] = self.chain_embedding
        # ## defining ese layers
        # if ese_vec_embedding_layer:
        #     for feature in ['user_prev_chains_ese', 'freq_chains_ese', 'freq_clicks_ese']:
        #         self.feat_to_partial_tower[feature] = ese_vec_embedding_layer

        # define order source embedding
        # if unique_order_sources is not None and len(unique_order_sources) > 1\
        #     and 'account_order_source' in query_features:
        #
        #     self.order_source_tower = create_string_embedding_layer(layer_prefix='order_source',
        #                                                             vocab=unique_order_sources,
        #                                                             embedding_size=chain_emb_size)
        #
        #     self.feat_to_partial_tower['account_order_source'] = self.order_source_tower

        delivery_area_feature_name = 'delivery_area_id'
        if delivery_area_feature_name in query_features and unique_area_ids is not None:
            embedding_layer = create_int_embedding_layer(
                layer_prefix=delivery_area_feature_name,
                vocab=unique_area_ids,
                embedding_size=16
            )
            self.delivery_area_id_tower = embedding_layer
            self.feat_to_partial_tower[delivery_area_feature_name] = self.delivery_area_id_tower

        for feature_col, vocab_size in zip(['order_hour', 'order_weekday'], [24, 7]):

            if feature_col in query_features:
                embedding_layer = create_int_embedding_layer(
                    layer_prefix=feature_col,
                    vocab=np.arange(vocab_size),
                    embedding_size=16
                )
                setattr(self, feature_col + '_tower', embedding_layer)
                self.feat_to_partial_tower[feature_col] = embedding_layer

        self.numeric_features = []
        self.query_features = query_features
        for chain_feature in self.query_features:
            num_flags = ['log', 'avg', 'pct', 'eur', 'cnt', 'tpro']
            for flag in num_flags:
                if flag in chain_feature:
                    self.numeric_features.append(chain_feature)
                    break
        self.numeric_features = sorted(self.numeric_features)
        num_features_length = len(self.numeric_features)
        if num_features_length > 0:
            for numeric_feature in self.numeric_features:
                numeric_features_input_layer = tf.keras.layers.Input(
                    shape=(1), name=numeric_feature + '_input'
                )
                numeric_feature_input = tf.keras.layers.Lambda(lambda x: tf.reshape(
                    # numeric_features_input_layer(x), (-1, 1)
                    tf.cast(x, tf.float32), (-1, 1)
                ))
                self.feat_to_partial_tower[numeric_feature] = tf.keras.Sequential(
                    [numeric_features_input_layer, numeric_feature_input])

    def call(self, features_df):
        potential_features = self.numeric_features + [
            "freq_items",
            "prev_items",
            "prev_items_names",
            "freq_items_names",
            "chain_id",
            "order_hour",
            "order_weekday",
            'delivery_area_id',
            'chain_prev_items',
            'chain_prev_items_names'
        ]
        feature_col_dict = {x: x for x in potential_features}
        # feature_col_dict['freq_clicks_ese'] = 'freq_clicks'
        # feature_col_dict['freq_chains_ese'] = 'freq_chains'
        # feature_col_dict['user_prev_chains_ese'] = 'user_prev_chains'
        print('tracing')
        if len(self.feat_to_partial_tower) > 1:
            available_features = [feature for feature in self.feat_to_partial_tower.keys()]
            output = tf.concat(
                [
                    self.feat_to_partial_tower[feature](features_df[feature_col_dict[feature]])
                    for feature in available_features if feature in self.feat_to_partial_tower
                ],
                axis=1,
            )

        return output


class itemModel_v1(tf.keras.Model):
    def __init__(
            self,
            item_emb_size,
            unique_chain_ids,
            unique_item_ids,
            unique_items_vocab,
            cuisine_emb_size,
            candidates_features,
            aggregation,
            shared_item_embeddings=None,
            shared_item_vocab_embeddings=None,
    ):
        super().__init__()

        self.candidates_features = candidates_features
        self.feat_to_partial_tower = {}
        if 'item_id' in candidates_features:
            if shared_item_embeddings:
                embedding = shared_item_embeddings
            else:
                embedding = tf.keras.layers.Embedding(
                    len(unique_item_ids) + 1, item_emb_size, name="item_embedding"
                )
            self.item_embedding = tf.keras.Sequential(
                [
                    tf.keras.layers.StringLookup(
                        vocabulary=unique_item_ids, mask_token="0", name="item_lookup"
                    ),
                    embedding,
                ],
                name='item_id' + '_layer'
            )
            self.feat_to_partial_tower["item_id"] = self.item_embedding

        if 'chain_id' in candidates_features:
            embedding = tf.keras.layers.Embedding(
                len(unique_chain_ids) + 1, item_emb_size, name="chain_embedding"
            )
            self.chain_embedding = tf.keras.Sequential(
                [
                    tf.keras.layers.StringLookup(
                        vocabulary=unique_chain_ids, mask_token=None, name="chain_lookup"
                    ),
                    embedding,
                ],
                name='chain_id' + '_layer'
            )
            self.feat_to_partial_tower["chain_id"] = self.chain_embedding

        # if ese_ch_embedding_layer:
        #     print('created layer')
        #     self.feat_to_partial_tower["chain_ese_embedding"] = ese_ch_embedding_layer
        # else:
        #     print('not created')

        # if chain_cuisine_feat:
        #     self.cuisine_vectorizer = tf.keras.layers.TextVectorization(
        #         vocabulary=list(unique_cuisines_names),
        #         standardize="lower_and_strip_punctuation",
        #     )
        #
        #     self.cuisine_text_embedding = tf.keras.Sequential(
        #         [
        #             self.cuisine_vectorizer,
        #             tf.keras.layers.Embedding(
        #                 len(unique_cuisines_names) + 2, cuisine_emb_size, mask_zero=True
        #             ),
        #             tf.keras.layers.GlobalAveragePooling1D(),
        #         ]
        #     )
        #     self.feat_to_partial_tower["tlabel"] = self.cuisine_text_embedding
        # self.chain_cuisine_feat = chain_cuisine_feat

        self.numeric_features = []
        for chain_feature in self.candidates_features:
            num_flags = ['log', 'avg', 'pct', 'eur', 'cnt']
            for flag in num_flags:
                if flag in chain_feature:
                    self.numeric_features.append(chain_feature)
                    break
        self.numeric_features = sorted(self.numeric_features)
        num_features_length = len(self.numeric_features)
        if num_features_length > 0:
            for numeric_feature in self.numeric_features:
                numeric_features_input_layer = tf.keras.layers.Input(
                    shape=(1), name=numeric_feature + '_input'
                )
                numeric_feature_input = tf.keras.layers.Lambda(lambda x: tf.reshape(
                    # numeric_features_input_layer(x), (-1, 1)
                    tf.cast(x, tf.float32), (-1, 1)
                ))
                self.feat_to_partial_tower[numeric_feature] = tf.keras.Sequential(
                    [numeric_features_input_layer, numeric_feature_input])

        for feature in ['chain_name', 'item_name', 'item_description_en']:
            vocab = unique_items_vocab
            shared_embedding = shared_item_vocab_embeddings
            if feature in candidates_features:
                feature_tower = create_vectorized_attention_layer(
                    vocab=list(vocab), aggregation=aggregation,
                    emebdding_size=item_emb_size, attention_dim=item_emb_size,
                    layer_name_prefix=feature, predefined_embedding=shared_embedding
                )
                setattr(self, feature + '_tower', feature_tower)

                self.feat_to_partial_tower[feature] = feature_tower
    @tf.function
    def call(self, chains_df):
        potential_features = self.numeric_features + [
            "chain_id",
            "item_id",
            "item_description_en",
            "chain_name",
            "item_name"
        ]

        feature_col_dict = {x: x for x in potential_features}
        # feature_col_dict['chain_ese_embedding'] = 'chain_id'

        available_features = [
            feature for feature in potential_features if feature in chains_df
        ]  # + ['chain_ese_embedding']

        output = tf.concat(
            [
                self.feat_to_partial_tower[feature](chains_df[feature_col_dict[feature]])
                for feature in available_features
            ],
            axis=1,
        )

        return output


class AccountItemModel(tfrs.Model):
    def __init__(
            self,
            customer_model,
            item_model,
            candidates_ds,
            query_features,
            candidate_features,
            label_smoothing=0.0,
            temperature=1,
            unique_candidate="item_id",
            num_hard_negatives=None,
            unique_candidates_features=None,
            batch_size=0
    ):
        super().__init__()
        self.item_model: tf.keras.Model = item_model
        self.customer_model: tf.keras.Model = customer_model

        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidates_ds.batch(128).map(self.item_model),
                ks=(1,)
            ),
            remove_accidental_hits=True,
            temperature=temperature,
            num_hard_negatives=num_hard_negatives
            # loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
        )
        self.query_features = query_features
        self.candidate_features = candidate_features
        self.unique_candidate = unique_candidate
        self.unique_candidate_features = unique_candidates_features
        self.batch_size = batch_size

    def compute_loss(self, features, training=False):
        mns = features[1]
        features = features[0]
        customer_embeddings = self.customer_model(
            {a: features[a] for a in self.query_features}, training=True
        )
        item_embeddings = self.item_model(
            {
                a: tf.concat([features[a], mns[a]], axis=0)
                for a in self.candidate_features
            },
            training=True,
        )
        # if 'candidate_probability' in features.columns:
        #    candidate_probability = features['candidate_probability']

        ## mixed negative sampling
        candidate_probability = tf.concat(
            [features["candidate_probability"], mns["candidate_probability"]], axis=0
        )
        candidate_ids = tf.concat(
            [features[self.unique_candidate], mns[self.unique_candidate]], axis=0
        )
        sample_weight = features["sample_weight"]

        if self.unique_candidate_features is not None:
            uniform_negatives = self.unique_candidate_features.sample(self.batch_size)
            negatives_embeddings = self.item_model(uniform_negatives)
            item_embeddings = tf.concat(
                [item_embeddings, negatives_embeddings], axis=0
            )
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

        return self.task(
            customer_embeddings,
            item_embeddings,
            compute_metrics=not training,
            candidate_sampling_probability=candidate_probability,
            candidate_ids=candidate_ids,
            sample_weight=sample_weight,
        )


def create_item_two_tower_model(
        embedding_dimension,
        unique_chain_ids,
        unique_items_ids,
        unique_items_vocab,
        unique_area_ids,
        candidates_ds,
        query_features=[],
        candidate_features=[],
        temperature=1,
        label_smoothing=0,
        unique_candidate="item_id",
        aggregation="AVG",
        num_hard_negatives=None,
        unique_candidates_features=None,  # for negative sampling
        batch_size=0,
        dropout=0,
        enable_shared_item_embeddings=False,
        enable_shared_keywords_embedding=False,
        chain_features_df=None
):
    if enable_shared_item_embeddings:
        shared_item_embeddings = tf.keras.layers.Embedding(
            len(unique_items_ids) + 2,
            embedding_dimension,
            name="shared_embedding",
            mask_zero=True,
        )
    else:
        shared_item_embeddings = None
    if enable_shared_keywords_embedding:
        shared_item_content_embedding = tf.keras.layers.Embedding(
            len(unique_items_vocab) + 2,
            embedding_dimension,
            name="keyword_embedding",
            mask_zero=True,
        )
    else:
        shared_item_content_embedding = None

    normalization = tf.keras.layers.Lambda(
        lambda x: tf.keras.backend.l2_normalize(x, axis=-1)
    )
    customer_model = tf.keras.Sequential(
        [
            userModel_v1(
                item_emb_size=embedding_dimension,
                unique_chain_ids=unique_chain_ids,
                unique_item_ids=unique_items_ids,
                unique_items_vocab=unique_items_vocab,
                unique_area_ids=unique_area_ids,
                query_features=query_features,
                aggregation=aggregation,
                shared_item_embeddings=shared_item_embeddings,
                shared_item_vocab_embedding=shared_item_content_embedding
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
    item_model = tf.keras.Sequential(
        [
            itemModel_v1(
                item_emb_size=embedding_dimension,
                unique_chain_ids=unique_chain_ids,
                unique_item_ids=unique_items_ids,
                unique_items_vocab=unique_items_vocab,
                cuisine_emb_size=embedding_dimension,
                candidates_features=candidate_features,
                aggregation=aggregation,
                shared_item_embeddings=shared_item_embeddings,
                shared_item_vocab_embeddings=shared_item_content_embedding
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
    model = AccountItemModel(
        customer_model,
        item_model,
        candidates_ds=candidates_ds,
        query_features=query_features,
        candidate_features=candidate_features,
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


def create_tt_v1_menu_user_model_from_config(**params):
    if params['enable_shared_item_embeddings']:
        shared_item_embeddings = tf.keras.layers.Embedding(len(params['unique_item_ids']) + 2,
                                                           params['embedding_dimension'],
                                                           name='shared_embedding',
                                                           mask_zero=True)
    else:
        shared_item_embeddings = None
    if params['enable_shared_keywords_embedding']:
        shared_items_content_embedding = tf.keras.layers.Embedding(len(params['unique_items_vocab']) + 2,
                                                                   params['embedding_dimension'],
                                                                   name='keyword_embedding',
                                                                   mask_zero=True)
    else:
        shared_items_content_embedding = None

    normalization = tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=-1))
    last_layer_dimension = params.get('last_layer_dimension', 32)
    customer_model = tf.keras.Sequential([
        userModel_v1(
            item_emb_size=params['embedding_dimension'],
            unique_chain_ids=params['unique_chain_ids'],
            unique_item_ids=params['unique_item_ids'],
            unique_items_vocab=params['unique_items_vocab'],
            unique_area_ids=params['unique_area_ids'],
            query_features=params['query_features'],
            aggregation=params['aggregation'],
            shared_item_embeddings=shared_item_embeddings,
            shared_item_vocab_embedding=shared_items_content_embedding
        ),
        tf.keras.layers.BatchNormalization(),
        tfrs.layers.dcn.Cross(kernel_initializer="glorot_uniform"),
        tf.keras.layers.Dense(params['embedding_dimension'] * 4, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(params['embedding_dimension'] * 2, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(last_layer_dimension),
        normalization,
    ], name='query_tower')

    return customer_model


def create_tt_v1_menu_item_model_from_config(**params):
    if params['enable_shared_item_embeddings']:
        shared_item_embeddings = tf.keras.layers.Embedding(len(params['unique_item_ids']) + 2,
                                                           params['embedding_dimension'],
                                                           name='shared_embedding',
                                                           mask_zero=True)
    else:
        shared_item_embeddings = None
    if params['enable_shared_keywords_embedding']:
        shared_items_content_embedding = tf.keras.layers.Embedding(len(params['unique_items_vocab']) + 2,
                                                                   params['embedding_dimension'],
                                                                   name='keyword_embedding',
                                                                   mask_zero=True)
    else:
        shared_items_content_embedding = None
    last_layer_dimension = params.get('last_layer_dimension', 32)
    normalization = tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=-1))
    item_model = tf.keras.Sequential(
        [
            itemModel_v1(
                item_emb_size=params['embedding_dimension'],
                unique_chain_ids=params['unique_chain_ids'],
                unique_item_ids=params['unique_item_ids'],
                unique_items_vocab=params['unique_items_vocab'],
                cuisine_emb_size=params['embedding_dimension'],
                candidates_features=params['candidate_features'],
                aggregation=params['aggregation'],
                shared_item_embeddings=shared_item_embeddings,
                shared_item_vocab_embeddings=shared_items_content_embedding
            ),
            tf.keras.layers.BatchNormalization(),
            tfrs.layers.dcn.Cross(kernel_initializer="glorot_uniform"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(params['embedding_dimension'] * 2, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(last_layer_dimension),
            normalization,
        ],
        name="candidate_tower",
    )

    return item_model

