import os
import mlflow
import argparse
import warnings
import structlog
from dotenv import load_dotenv
from datetime import timedelta

import pandas as pd
from datetime import datetime
import numpy as np
import ast

from projects.vendor_ranking.common.two_towers.src.utils.bq_client import (
    write_to_bq,
    delete_data_for_date,
    create_clustered_partitioned_table,
    update_timestamp_column
)

from projects.vendor_ranking.common.two_towers.src.data.processors.feast import (
    ChainFeaturesProcessor
)
from projects.vendor_ranking.common.two_towers.src.cli.prepare import (
    CHAIN_FEATURE_COLUMNS
)
from projects.vendor_ranking.common.two_towers.src.data import (
    query_loader,
    get_data_fetcher,
    get_feature_store
)
from projects.vendor_ranking.common.two_towers.evaluation import (
    load_mlflow_model
)
from projects.vendor_ranking.two_towers_v1 import (
    PROD_MODEL_ALIAS
)

warnings.simplefilter(action='ignore', category=FutureWarning)
load_dotenv(override=True)
LOG: structlog.stdlib.BoundLogger = structlog.get_logger()

ESE_SIM_THRESHOLD = 0.7
RATING_COUNT_THRESHOLD = 100
RATING_THRESHOLD = 4.0
IMPRESSIONS_THRESHOLD = 15000
NEW_CHAIN_TIMEFRAME = 60
INCLUDED_VENDOR_GRADES = ['A', 'B']


CURRENT_DATE_MINUS_ONE = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
CURRENT_DATE_MINUS_THIRTY = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
DATE_TO_WRITE = datetime.now() - timedelta(days=1)
# CURRENT_DATE_MINUS_ONE = '2024-09-08'
# CURRENT_DATE_MINUS_THIRTY = '2024-09-01'
# DATE_TO_WRITE = datetime.now() - timedelta(days=1)


def download_chain_features(country, candidate_features):
    chain_features_query = query_loader.load_query(
        'chain_features.sql.j2',
        start_date=CURRENT_DATE_MINUS_THIRTY,
        end_date=CURRENT_DATE_MINUS_ONE,
        country_code=country,
        orders_profile_table=get_feature_store().get_data_source("orders_profile").get_table_query_string(),
        chain_profile_table=get_feature_store().get_data_source("chain_profile").get_table_query_string()
    )
    df_chain_features = get_data_fetcher().fetch_data(
        description='chain features',
        source="feast",
        entity_sql=chain_features_query,
        features=CHAIN_FEATURE_COLUMNS
    )
    chain_features_processor = ChainFeaturesProcessor(
        df_chain_features,
        candidate_features
    )
    chain_features = chain_features_processor.process()
    chain_features['chain_id'] = chain_features['chain_id'].astype(int)
    return chain_features


def vectorized_similarity(v1, v2):
    if len(v1) == len(v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        sim = (v1 * v2).sum() / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return sim
    return np.nan


def str2lst(s):
    try:
        return list(ast.literal_eval(s))
    except:
        return []


def read_query_from_file(file_path, **kwargs):
    with open(file_path, 'r') as file:
        query = file.read()

    for key, value in kwargs.items():
        print('{' + key + '}')
        print(str(value))
        query = query.replace('{' + key + '}', str(value))

    print("SQL Query with parameters:", query)
    return query


def download_chain_features_coldstart(country, **kwargs):
    chain_features_coldstart_query = query_loader.load_query(
        'chain_cold_start_data_fetch.sql.j2',
        country=country,
        IMPRESSIONS_THRESHOLD=IMPRESSIONS_THRESHOLD,
        NEW_CHAIN_TIMEFRAME=NEW_CHAIN_TIMEFRAME,
        INCLUDED_VENDOR_GRADES=tuple(INCLUDED_VENDOR_GRADES)
    )
    chain_features_coldstart = get_data_fetcher().fetch_data(
        description='Two tower Trained model chain embeddings',
        source="sql",
        query=chain_features_coldstart_query,
    )
    chain_features_coldstart['chain_id'] = chain_features_coldstart['chain_id'].astype(int)
    return chain_features_coldstart


def impute_popularity_features(raw_chain_features_df, coldstart_chain_features_df):
    df_chain_city = pd.merge(raw_chain_features_df, coldstart_chain_features_df[['chain_id', 'chain_type', 'city_id']]
                             , on='chain_id', how='left')
    df_chain_city = df_chain_city.explode('city_id')
    df_city_popularity = df_chain_city[df_chain_city['chain_type'] == 'existing chain'].groupby(['city_id']).agg(
        pop_chain_daily_orders_per_vendor_log=('chain_daily_orders_per_vendor_log', 'mean'),
        pop_chain_avg_daily_active_orders_log=('chain_avg_daily_active_orders_log', 'mean')
    ).reset_index()
    df_chain_city = pd.merge(df_chain_city, df_city_popularity[
        ['city_id', 'pop_chain_daily_orders_per_vendor_log', 'pop_chain_avg_daily_active_orders_log']]
                             , how='left', on='city_id')
    df_new_chain_popularity_values = df_chain_city[df_chain_city['chain_type'] == 'new chain'].groupby('chain_id').agg(
        pop_chain_daily_orders_per_vendor_log=('pop_chain_daily_orders_per_vendor_log', 'mean'),
        pop_chain_avg_daily_active_orders_log=('pop_chain_avg_daily_active_orders_log', 'mean')
    ).reset_index()
    chain_features_df = pd.merge(raw_chain_features_df, df_new_chain_popularity_values, how='left', on='chain_id')

    LOG.info(f"Features for {len(df_new_chain_popularity_values)} cold start chains found")
    LOG.info(f"""Popularity features imputed for {(chain_features_df['pop_chain_daily_orders_per_vendor_log'] >
                                                   chain_features_df['chain_daily_orders_per_vendor_log']).sum()
    } cold start chains""")
    chain_features_df['chain_daily_orders_per_vendor_log'] = np.where(
        chain_features_df['pop_chain_daily_orders_per_vendor_log'] >
        chain_features_df['chain_daily_orders_per_vendor_log'],
        chain_features_df['pop_chain_daily_orders_per_vendor_log'],
        chain_features_df['chain_daily_orders_per_vendor_log'])

    chain_features_df['chain_avg_daily_active_orders_log'] = np.where(
        chain_features_df['pop_chain_avg_daily_active_orders_log'] >
        chain_features_df['chain_avg_daily_active_orders_log'],
        chain_features_df['pop_chain_avg_daily_active_orders_log'],
        chain_features_df['chain_avg_daily_active_orders_log'])

    chain_features_df.drop(columns=['pop_chain_daily_orders_per_vendor_log', 'pop_chain_avg_daily_active_orders_log'],
                           inplace=True)
    return chain_features_df


def get_cold_start_embeddings(chain_embeddings_df, chain_features_df, coldstart_chain_features_df):
    df_missing_chains = coldstart_chain_features_df[(coldstart_chain_features_df['chain_type'] == 'new chain') &
                                                    (~coldstart_chain_features_df['chain_id'].isin(
                                                        chain_features_df['chain_id'])) &
                                                    (coldstart_chain_features_df['ese_embeddings'].notnull())].copy()

    df_candidate_chains = pd.merge(chain_features_df, coldstart_chain_features_df[['chain_id', 'chain_type', 'city_id',
                                                                                   'ese_embeddings', 'rating',
                                                                                   'rating_count']], on='chain_id',
                                   how='left')
    df_candidate_chains = df_candidate_chains[(df_candidate_chains['chain_type'] == "existing chain") &
                                              (df_candidate_chains['rating_count'] >= RATING_COUNT_THRESHOLD) &
                                              (df_candidate_chains['rating'] >= RATING_THRESHOLD)]
    df_candidate_chains = df_candidate_chains[df_candidate_chains['ese_embeddings'].notnull()]

    if len(df_missing_chains) == 0 or len(df_candidate_chains) == 0:
        return chain_embeddings_df

    df_missing_chains = df_missing_chains.explode('city_id')[['chain_id', 'ese_embeddings', 'city_id']]

    df_candidate_chains = df_candidate_chains.explode('city_id')[['chain_id', 'ese_embeddings', 'city_id']]
    df_candidate_chains.columns = ['candidate_chain_id', 'candidate_ese_embeddings', 'city_id']

    df_similar_chains = pd.merge(df_missing_chains, df_candidate_chains, how='inner', on='city_id')
    df_similar_chains['similarity'] = df_similar_chains.apply(
        lambda x: vectorized_similarity(str2lst(x['ese_embeddings']), str2lst(x['candidate_ese_embeddings'])), axis=1)
    df_similar_chains = df_similar_chains[df_similar_chains['similarity'] >= ESE_SIM_THRESHOLD]
    df_similar_chains = pd.merge(df_similar_chains, chain_embeddings_df, how='inner', left_on='candidate_chain_id',
                                 right_on='chain_id')
    df_similar_chains.drop(columns=['chain_id_y'], inplace=True)
    df_similar_chains.rename(columns={'chain_id_x': 'chain_id'}, inplace=True)
    df_similar_chains['embeddings'] = df_similar_chains['embeddings'].apply(lambda x: str2lst(x))

    new_chain_embedding = {}
    for i, new_chain in enumerate(df_similar_chains['chain_id'].unique()):
        data = df_similar_chains[df_similar_chains['chain_id'] == new_chain]
        if data.shape[0] > 0:
            new_chain_embedding[new_chain] = np.mean(data['embeddings'].to_list(), axis=0)
    df_new_chains = pd.DataFrame(new_chain_embedding.items(), columns=['chain_id', 'embeddings'])
    df_new_chains['embeddings'] = df_new_chains['embeddings'].apply(list)
    df_new_chains['embeddings'] = df_new_chains['embeddings'].astype('str')
    LOG.info(f"Embeddings added for {len(df_new_chains)} cold start chains")
    return pd.concat((chain_embeddings_df, df_new_chains), axis=0)


def compute_chain_embeddings(country, dst_chain_embeddings_table, model):
    raw_chain_features_df = download_chain_features(
        country,
        model.two_tower_model.params.get("candidate_features")
    )
    coldstart_chain_features_df = download_chain_features_coldstart(country)
    chain_features_df = impute_popularity_features(
        raw_chain_features_df,
        coldstart_chain_features_df
    )

    chain_features_cols = model.two_tower_model.params.get("candidate_features")
    chain_features_df['chain_id'] = chain_features_df['chain_id'].astype(str)
    chain_embeddings = model.two_tower_model.chain_model(
        chain_features_df[chain_features_cols]
    ).numpy()

    chain_ids = chain_features_df["chain_id"].tolist()

    assert np.isnan(chain_embeddings).sum() == 0
    chain_embeddings_df = pd.DataFrame(
        {"chain_id": chain_ids, "embeddings": chain_embeddings.tolist()}
    )

    chain_embeddings_df['chain_id'] = chain_embeddings_df['chain_id'].astype('int')
    chain_embeddings_df['embeddings'] = chain_embeddings_df['embeddings'].astype('str')

    chain_features_df['chain_id'] = chain_features_df['chain_id'].astype('int')
    all_chain_embeddings_df = get_cold_start_embeddings(
        chain_embeddings_df,
        chain_features_df,
        coldstart_chain_features_df
    )
    all_chain_embeddings_df.embeddings = all_chain_embeddings_df.embeddings.apply(
        lambda x: x.replace("[", "{").replace("]", "}").replace("'", '"')
    )
    all_chain_embeddings_df["feature_timestamp"] = DATE_TO_WRITE
    all_chain_embeddings_df["country_code"] = country

    all_chain_embeddings_df = all_chain_embeddings_df[
        ["chain_id", "embeddings", "feature_timestamp", "country_code"]
    ]

    delete_data_for_date(dst_chain_embeddings_table, DATE_TO_WRITE, country)
    write_to_bq(df=all_chain_embeddings_df, table_name=dst_chain_embeddings_table, if_exists="append")
    update_timestamp_column(DATE_TO_WRITE.strftime('%Y-%m-%d'), dst_chain_embeddings_table, country)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cold Start chain embedding"
    )
    parser.add_argument(
        "--country",
        required=True,
        choices=['EG', 'AE', 'KW', 'QA', 'OM', 'BH', 'IQ', 'JO'],
        help="Possible country values are: ['EG', 'AE', 'KW', 'QA', 'OM', 'BH', 'IQ', 'JO']"
    )
    args = parser.parse_args()

    DST_TABLE_NAME = f"data_feature_store.chain_embeddings_for_two_tower_coldstart_v3_mlf_all"
    DATASET_ID = "data_feature_store"
    TABLE_NAME = "chain_embeddings_for_two_tower_coldstart_v3_mlf_all"
    KEY_ENTITY_FIELD = "chain_id"
    KEY_ENTITY_TYPE = "integer"
    FIELDS = ["chain_id", "feature_timestamp", "embeddings", "country_code"]
    create_clustered_partitioned_table(
        DATASET_ID, TABLE_NAME, KEY_ENTITY_FIELD, KEY_ENTITY_TYPE, FIELDS
    )

    exp_name = os.environ['MLFLOW_EXPERIMENT_NAME']
    model_name = f"{exp_name}_{args.country.lower()}"
    LOG.info(f"Tracking uri: {mlflow.get_tracking_uri()}")
    LOG.info(f"Experiment id: {exp_name}")
    LOG.info(f"MODEL_NAME: {model_name}")

    model = load_mlflow_model(
        model_name,
        PROD_MODEL_ALIAS
    )

    LOG.info(
        f"Computing embeddings for {args.country}"
    )

    compute_chain_embeddings(
        args.country,
        DST_TABLE_NAME,
        model
    )

    LOG.info(f"Embeddings for {args.country} computed successfully!")
