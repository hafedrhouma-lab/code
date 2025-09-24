import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
from typing import TYPE_CHECKING

from ..utils.eval_utils import df_to_np_embeddings


if TYPE_CHECKING:
    from pandas import DataFrame


def create_ese_embedding_layer(embeddings_df, is_vector) -> "tf.keras.Sequential":
    embeddings_np = df_to_np_embeddings(embeddings_df)
    embeddings_df.index = embeddings_df.index.astype("str")
    # UNK embedding
    embeddings_mean = embeddings_np.mean(axis=0).reshape((1, -1))
    # Mask Embedding
    zeros = np.zeros(embeddings_mean.shape)
    ese_vocab = embeddings_df.index.values
    ese_vectorizer = tf.keras.layers.TextVectorization(
        vocabulary=list(ese_vocab), standardize="strip_punctuation"
    )
    ese_agg = tf.keras.layers.GlobalAveragePooling1D()

    if is_vector:
        vectorizer_embedding_np = np.concatenate([zeros, embeddings_mean, embeddings_np], axis=0)
        vocab_length = len(ese_vocab) + 2
        mask_zero = True
        layer_name = "ese_vector_embedding"
    else:
        vectorizer_embedding_np = np.concatenate([zeros, embeddings_mean, embeddings_np], axis=0)
        vocab_length = len(ese_vocab) + 2
        mask_zero = True
        layer_name = "ese_chain_embedding"
    ese_embedding_size = vectorizer_embedding_np.shape[1]
    ## 2 for unk + mask in batching
    ese_embedding = tf.keras.layers.Embedding(
        input_dim=vocab_length, output_dim=ese_embedding_size,
        mask_zero=mask_zero, trainable=False, weights=[vectorizer_embedding_np]
    )
    ese_vector_embedding_layer = tf.keras.Sequential(
        [
            ese_vectorizer,
            ese_embedding,
            ese_agg
        ],
        name=layer_name
    )
    if is_vector:
        test_vocab = embeddings_df.index.values
        test_vocab = " ".join(test_vocab[1:4])
        ese_vector_embedding_layer([test_vocab])
    else:
        test_vocab = embeddings_df.index.values[0]
        ese_vector_embedding_layer([test_vocab])
    assert len(ese_vector_embedding_layer.layers[1].weights) > 0
    return ese_vector_embedding_layer


def get_ese_embedding_tf_layer(is_vector: bool, embeddings_df: "DataFrame"
) -> "tf.keras.Sequential":
    ese_vector_embedding_layer = create_ese_embedding_layer(embeddings_df, is_vector)
    return ese_vector_embedding_layer
