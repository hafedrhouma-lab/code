import tensorflow as tf

from abstract_ranking.two_tower.transformer_encoder_layer import GlobalSelfAttention


def create_vectorized_attention_layer(
    vocab,
    aggregation,
    embedding_size,
    attention_dim,
    layer_name_prefix,
    predefined_embedding=None,
    dropout=0.1,
    apply_attention=True
):
    vectorizer = tf.keras.layers.TextVectorization(
        vocabulary=list(vocab),
        standardize="strip_punctuation",
        name=layer_name_prefix + "_vectorizer",
    )
    if aggregation == "AVG":
        chains_agg = tf.keras.layers.GlobalAveragePooling1D(name=layer_name_prefix + "_avg_pooling")
    else:
        chains_agg = tf.keras.layers.GRU(units=embedding_size)
    if predefined_embedding:
        embedding = predefined_embedding
    else:
        embedding = tf.keras.layers.Embedding(
            len(vocab) + 2,
            embedding_size,
            mask_zero=True,
            name=layer_name_prefix + "_embedding",
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
        layers,
        name=layer_name_prefix + "_tower",
    )
    return vectorized_embedding


def create_int_embedding_layer(layer_prefix, vocab, embedding_size):
    embedding_layer = tf.keras.Sequential(
        [
            tf.keras.layers.IntegerLookup(
                vocabulary=vocab, mask_token=None, name=layer_prefix + "_lookup"
            ),
            tf.keras.layers.Embedding(len(vocab) + 1, embedding_size, name=layer_prefix + "_embedding"),
        ],
        name=layer_prefix + "_layer",
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
