import os
import tensorflow as tf
import numpy as np
from vendor_ranking.two_tower.data_utils import df_to_np_embeddings
from vendor_ranking.two_tower.inference_ese import get_ese_chain_embeddings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def download_ese_embeddings(embedding_date, country_code):
    embeddings_df = get_ese_chain_embeddings(embedding_date, country_code=country_code)
    return embeddings_df


def create_ese_embedding_layer(embeddings_df, is_vector):
    embeddings_np = df_to_np_embeddings(embeddings_df)
    embeddings_df.index = embeddings_df.index.astype("str")
    ## UNK embedding
    embeddings_mean = embeddings_np.mean(axis=0).reshape((1, -1))
    ## Mask Embedding
    zeros = np.zeros(embeddings_mean.shape)
    ese_vocab = embeddings_df.index.values
    ese_vectorizer = tf.keras.layers.TextVectorization(vocabulary=list(ese_vocab), standardize="strip_punctuation")
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
        input_dim=vocab_length,
        output_dim=ese_embedding_size,
        mask_zero=mask_zero,
        trainable=False,
        weights=[vectorizer_embedding_np],
    )
    ese_vector_embedding_layer = tf.keras.Sequential([ese_vectorizer, ese_embedding, ese_agg], name=layer_name)
    if is_vector:
        test_vocab = embeddings_df.index.values
        test_vocab = " ".join(test_vocab[1:4])
        ese_vector_embedding_layer([test_vocab])
    else:
        test_vocab = embeddings_df.index.values[0]
        ese_vector_embedding_layer([test_vocab])
    assert len(ese_vector_embedding_layer.layers[1].weights) > 0
    return ese_vector_embedding_layer


def get_ese_embedding_tf_layer(embedding_date, country_code, is_vector, embeddings_df=None):
    if embeddings_df is None:
        embeddings_df = get_ese_chain_embeddings(embedding_date, country_code=country_code)
    ese_vector_embedding_layer = create_ese_embedding_layer(embeddings_df, is_vector)
    return ese_vector_embedding_layer


test = False
if test:
    embeddings_df = get_ese_chain_embeddings("2023-06-01", country_code="AE")
    embeddings_df.index = embeddings_df.index.astype("str")
    ese_vector_embedding_layer, a, b = create_ese_embedding_layer(embeddings_df)
    ese_vocab = list(embeddings_df.index.values)

    print(ese_vocab[0:2] + ese_vocab[5:7])
    sentence = " ".join(ese_vocab[0:2] + ese_vocab[5:7])
    selected_embeddings = df_to_np_embeddings(embeddings_df.loc[ese_vocab[0:2] + ese_vocab[5:7]])
    print(selected_embeddings.shape)
    manual_avg = selected_embeddings.mean(axis=0)
    emb_avg = ese_vector_embedding_layer([sentence])
    print("first_layer", a([sentence]))
    first_layer_output = a([sentence])
    print("second_layer", b(first_layer_output))
    print("last_layer", ese_vector_embedding_layer([sentence, sentence]))
    # print(manual_avg, emb_avg)
