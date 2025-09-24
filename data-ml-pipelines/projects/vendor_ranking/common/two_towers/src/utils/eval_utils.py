import pandas as pd
import numpy as np

from .md5_utils import get_md5_series_from_dataframe


def get_similar_geohashes(geohash_chains_df):
    geohash_chains_df = geohash_chains_df[
        ["address_geohash", "chain_id"]
    ].drop_duplicates()
    geohash_to_ch_list = geohash_chains_df.groupby("address_geohash").chain_id.apply(
        lambda x: set(x.tolist())
    )

    geohash_to_ch_list = geohash_to_ch_list.reset_index().rename(
        columns={"chain_id": "available_chains"}
    )

    geohash_to_ch_list["chains_md5"] = get_md5_series_from_dataframe(
        geohash_to_ch_list[["available_chains"]]
    )
    min_chain_per_geo = geohash_to_ch_list.groupby("chains_md5").address_geohash.min()
    min_chain_per_geo = min_chain_per_geo.reset_index().rename(
        columns={"address_geohash": "min_geo"}
    )
    geohash_to_ch_list = geohash_to_ch_list.merge(min_chain_per_geo, on="chains_md5")
    geohash_to_parent_geohash = {
        a: b
        for a, b in zip(geohash_to_ch_list.address_geohash, geohash_to_ch_list.min_geo)
    }
    geohash_to_ch_list = geohash_to_ch_list[
        geohash_to_ch_list["min_geo"] == geohash_to_ch_list["address_geohash"]
        ]
    geohash_to_chains = {
        a: b
        for a, b in zip(geohash_to_ch_list.min_geo, geohash_to_ch_list.available_chains)
    }

    return geohash_to_chains, geohash_to_parent_geohash


def df_to_np_embeddings(embeddings_df):
    return np.array([np.array([x for x in y]) for y in embeddings_df.values]).squeeze()


def get_embeddings_similarity(query_embedding, candidate_embeddings):
    sim_score = query_embedding.dot(candidate_embeddings.T)
    sim_score = (
            sim_score
            / np.linalg.norm(query_embedding)
            / np.linalg.norm(candidate_embeddings, axis=1)
    )
    return sim_score


def remove_redundant_values(values):
    df = pd.DataFrame(values, columns=["x"])
    df_keep_first = list(df.drop_duplicates(keep="first")["x"])
    return df_keep_first


def list_intersection_count(candidates, gt):
    return sum([1 for x in gt if x in set(candidates)])


def weighted_recall(candidates, prev_orders):
    return list_intersection_count(candidates, prev_orders) / len(prev_orders)
