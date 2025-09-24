import os

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ndcg_score,
    coverage_error,
)
import matplotlib.pyplot as plt

from vendor_ranking.two_tower.utils import date_to_str, read_query


def get_orders_df(start_date, end_date, country_code="AE", city_id=35):
    filename = f"orders_{date_to_str(start_date)}_{date_to_str(end_date)}_{city_id}_{country_code}.parquet"
    if os.path.exists(filename):
        orders_df = pd.read_parquet(filename)
    else:
        orders_query = f"""
            select account_id, chain_id, order_id, order_time_utc, country_iso, vertical_class, vertical,
                   sub_vertical, jslibs.h3.ST_H3(customer_selected_location, 9),
                   ST_GEOHASH(f.customer_selected_location, 7) geohash
            from `tlb-data-prod.data_platform.fct_order` f
            join `tlb-data-prod.data_platform.dim_location` l on f.location_id = l.location_id
            where f.order_time_utc between '{start_date}' and '{end_date}' and f.country_iso = '{country_code}'
            AND l.city_id = {city_id} and account_id > 0 and not is_test AND f.customer_selected_location is not NULL
            And not is_guest and is_successful and vertical='food'
            """

        orders_df = read_query(orders_query)
        orders_df.to_parquet(filename)

    return orders_df


def df_to_np_embeddings(embeddings_df):
    return np.array([np.array([x for x in y]) for y in embeddings_df.values]).squeeze()


def get_embeddings_similarity(query_embedding, candidate_embeddings):
    sim_score = query_embedding.dot(candidate_embeddings.T)
    sim_score = sim_score / np.linalg.norm(query_embedding) / np.linalg.norm(candidate_embeddings, axis=1)
    return sim_score


def remove_redundant_values(values):
    df = pd.DataFrame(values, columns=["x"])
    df_keep_first = list(df.drop_duplicates(keep="first")["x"])
    return df_keep_first


def set_intersection_count(candidates, gt):
    return len(set(candidates) & set(gt))


def list_intersection_count(candidates, gt):
    return sum([1 for x in gt if x in set(candidates)])


def weighted_recall(candidates, prev_orders):
    return list_intersection_count(candidates, prev_orders) / len(prev_orders)


def uniform_recall(candidates, prev_orders):
    return set_intersection_count(candidates, prev_orders) / len(set(prev_orders))


def compute_ndcg(candidates, prev_chains):
    relevance = [[prev_chains.count(x) for x in candidates]]
    scores = [[x for x in range(len(candidates))][::-1]]

    return ndcg_score(relevance, scores)


def compute_coverage_error(candidates, prev_chains):
    relevance = [[1 if x in prev_chains else 0 for x in candidates]]
    scores = [[x for x in range(len(candidates))][::-1]]

    return coverage_error(relevance, scores)


def remove_low_frequency_items(df, col, min_value):
    return df[df.groupby(col)[col].transform("count").ge(min_value)]


def plot_overlapping_densities():
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Create the data
    rs = np.random.RandomState(1979)
    x = rs.randn(500)
    g = np.tile(list("ABCDEFGHIJ"), 50)
    df = pd.DataFrame(dict(x=x, g=g))
    m = df.g.map(ord)
    df["x"] += m

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-0.25, light=0.7)
    g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=0.5, palette=pal)

    # Draw the densities in a few steps
    g.map(
        sns.kdeplot,
        "x",
        bw_adjust=0.5,
        clip_on=False,
        fill=True,
        alpha=1,
        linewidth=1.5,
    )
    g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=0.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(
            0,
            0.2,
            label,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )

    g.map(label, "x")

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-0.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
