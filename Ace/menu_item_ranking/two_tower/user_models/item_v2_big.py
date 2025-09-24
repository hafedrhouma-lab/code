import numpy as np
import structlog
import tensorflow as tf
import tensorflow_recommenders as tfrs

from abstract_ranking.two_tower.transformer_encoder_layer import GlobalSelfAttention

LOG = structlog.getLogger(__name__)


def create_vectorized_attention_layer(
    vocab, aggregation,
    embedding_size,
    attention_dim,
    layer_name_prefix,
    predefined_embedding=None,
    dropout=0.1,
    apply_attention=True
):
    vectorizer = tf.keras.layers.TextVectorization(
        vocabulary=list(vocab), standardize="lower_and_strip_punctuation",
        name=layer_name_prefix + '_vectorizer'
    )
    if aggregation == "AVG":
        chains_agg = tf.keras.layers.GlobalAveragePooling1D(name=layer_name_prefix + '_avg_pooling')
    else:
        chains_agg = tf.keras.layers.GRU(units=embedding_size)
    if predefined_embedding:
        embedding = predefined_embedding
    else:
        embedding = tf.keras.layers.Embedding(
            len(vocab) + 2, embedding_size, mask_zero=True,
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


class userModel_v101_big(tf.keras.Model):
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

        for feature in ['prev_items', 'freq_items', 'prev_items_names', 'freq_items_names']:
            vocab = unique_item_ids if 'names' not in feature else unique_items_vocab
            shared_embedding = shared_item_embeddings if 'names' not in feature else shared_item_vocab_embedding
            if feature in query_features:
                feature_tower = create_vectorized_attention_layer(
                    vocab=list(vocab), aggregation=aggregation,
                    embedding_size=item_emb_size, attention_dim=item_emb_size,
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

        delivery_area_feature_name = "delivery_area_id"
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
                    tf.cast(x, tf.float32), (-1, 1)
                ))
                self.feat_to_partial_tower[numeric_feature] = tf.keras.Sequential(
                    [numeric_features_input_layer, numeric_feature_input]
                )

    @tf.function
    def call(self, features_df):
        potential_features = self.numeric_features + [
            "freq_items",
            "prev_items",
            "prev_items_names",
            "freq_items_names",
            "chain_id",
            "order_hour",
            "order_weekday",
            "delivery_area_id",
        ]
        feature_col_dict = {x: x for x in potential_features}
        if len(self.feat_to_partial_tower) > 1:
            available_features = [feature for feature in self.feat_to_partial_tower.keys()]
            output = tf.concat(
                [
                    self.feat_to_partial_tower[feature](features_df[feature_col_dict[feature]])
                    for feature in available_features if feature in self.feat_to_partial_tower
                ],
                axis=1,
            )
        else:
            raise ValueError("No available layers in the model")

        LOG.debug("Tracing")
        return output


def create_tt_v101_menu_user_model_from_config(**params):
    if params['enable_shared_item_embeddings']:
        shared_item_embeddings = tf.keras.layers.Embedding(
            len(params['unique_item_ids']) + 2,
            params['embedding_dimension'],
            name='shared_embedding',
            mask_zero=True
        )
    else:
        shared_item_embeddings = None
    if params['enable_shared_keywords_embedding']:
        shared_items_content_embedding = tf.keras.layers.Embedding(
            len(params['unique_items_vocab']) + 2,
            params['embedding_dimension'],
            name='keyword_embedding',
            mask_zero=True)
    else:
        shared_items_content_embedding = None

    normalization = tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=-1))
    customer_model = tf.keras.Sequential([
        userModel_v101_big(
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
        tf.keras.layers.Dense(32),
        normalization,
    ], name='query_tower')

    return customer_model
