import os
import pandas as pd
import mlflow
from pathlib import Path
from feast import FeatureStore

feature_store_relative_path = Path(
    __file__).parent.resolve() / "feature_store_offline.yaml"

store = FeatureStore(
    fs_yaml_file=feature_store_relative_path
)

from projects.vendor_ranking.hf_transformers.utils.bq_client import (
    upload_to_gcs,
    check_if_blob_exists,
    read_parquet_from_gcs
)

from projects.vendor_ranking.hf_transformers import (
    BUCKET_NAME,
    DESTINATION_BLOB_NAME
)


import structlog
LOG: structlog.stdlib.BoundLogger = structlog.get_logger()


def read_data_fs(start_date, end_date, country_code):
    filename = 'raw_data_fs_' + country_code + '_' + start_date + '_' + end_date + '.parquet'

    if os.path.exists(filename):
        data_df = pd.read_parquet(filename)
    else:

        entity_sql = f"""
            SELECT
                op.order_id,
                op.account_id,
                op.chain_id,
                op.country_code,
                op.feature_timestamp AS event_timestamp
            FROM {store.get_data_source("orders_profile").get_table_query_string()} op
            LEFT JOIN {store.get_data_source("chain_profile").get_table_query_string()} cp
            ON op.chain_id = cp.chain_id and op.country_code = cp.country_code and op.feature_timestamp = cp.feature_timestamp
            WHERE op.feature_timestamp BETWEEN "{start_date}" AND "{end_date}"
            AND op.country_code = "{country_code}"
            AND (vertical = 'food' or secondary_verticals like '%Food%')
        """

        data_df = store.get_historical_features(
            entity_df=entity_sql,
            features=[
                "order_profile_2t_v3_fv:delivery_area_id",
                "order_profile_2t_v3_fv:week_start",
                "order_profile_2t_v3_fv:geohash",
                "order_profile_2t_v3_fv:order_date",
                "order_profile_2t_v3_fv:order_time_utc",

                "account_orders_2t_v3_fv:most_recent_10_orders",  # user_prev_chains
                "account_orders_2t_v3_fv:frequent_chains",  # 'freq_chains'

                "account_performance_2t_v3_fv:account_log_order_cnt",
                "account_performance_2t_v3_fv:account_log_avg_gmv_eur",
                "account_performance_2t_v3_fv:account_incentives_pct",
                "account_performance_2t_v3_fv:account_is_tpro",
                "account_performance_2t_v3_fv:account_discovery_pct",

                "account_engagement_2t_v3_fv:most_recent_10_clicks_wo_orders",  # prev_clicks
                "account_engagement_2t_v3_fv:frequent_clicks",  # 'freq_clicks'
                "account_impressions_2t_v3_fv:frequent_neg_impressions"
            ],
        ).to_df()

        rename_dict = {
            "most_recent_10_orders": "prev_chains",
            "frequent_chains": "freq_chains",
            "most_recent_10_clicks_wo_orders": "prev_clicks",
            "frequent_clicks": "freq_clicks"
        }

        data_df.rename(columns=rename_dict, inplace=True)

        data_df['order_hour'] = data_df['order_time_utc'].dt.hour
        data_df['order_weekday'] = data_df['order_time_utc'].dt.dayofweek
        data_df['geohash6'] = data_df['geohash'].astype(str).str[:6]

        data_df["account_log_avg_gmv_eur"] = data_df["account_log_avg_gmv_eur"].astype("float32")
        data_df['chain_id'] = data_df['chain_id'].astype(str)
        data_df["delivery_area_id"] = data_df["delivery_area_id"].astype("string")
        data_df['delivery_area_id'] = data_df['delivery_area_id'].fillna('asdasdas')

        data_df['account_is_tpro'] = data_df['account_is_tpro'].fillna(0)
        data_df['account_discovery_pct'] = data_df['account_discovery_pct'].fillna(1)
        data_df['account_log_order_cnt'] = data_df['account_log_order_cnt'].fillna(0)
        data_df['account_log_avg_gmv_eur'] = data_df['account_log_avg_gmv_eur'].fillna(data_df['account_log_avg_gmv_eur'].mean())
        data_df['account_incentives_pct'] = data_df['account_incentives_pct'].fillna(data_df['account_incentives_pct'].mean())

        data_df['freq_chains'] = data_df['freq_chains'].fillna('no_frequent_orders')
        data_df['freq_clicks'] = data_df['freq_clicks'].fillna('no_frequent_clicks')
        data_df['prev_clicks'] = data_df['prev_clicks'].fillna('no_recent_clicks')
        data_df['prev_chains'] = data_df['prev_chains'].fillna('first_order')
        data_df['frequent_neg_impressions'] = data_df['frequent_neg_impressions'].fillna('no_negative_impressions')

        data_df.to_parquet(filename)

    LOG.info(f"Start date: {data_df['order_date'].min()}")
    LOG.info(f"End date: {data_df['order_date'].max()}")
    LOG.info(f"Data Shape: {data_df.shape}")

    data_df.drop(columns=['country_code', 'event_timestamp', 'order_time_utc', 'order_date', 'week_start'],inplace=True)

    return data_df

def read_data_gcs_fs(start_date, end_date, country_code, data_points):
    filename = 'raw_train_data_fs_' + country_code + '_' + start_date + '_' + end_date + '.parquet'
    blob = f"{DESTINATION_BLOB_NAME}/{country_code}/"
    print("loading FS data")

    if check_if_blob_exists(BUCKET_NAME, filename, blob):
        LOG.info(f"Data loading from {BUCKET_NAME}/{blob}/{filename}")
        data_df = read_parquet_from_gcs(BUCKET_NAME, filename, blob)

    else:
        entity_sql = f"""
            SELECT
                op.order_id,
                op.account_id,
                op.chain_id,
                op.country_code,
                op.feature_timestamp AS event_timestamp
            FROM {store.get_data_source("orders_profile").get_table_query_string()} op
            LEFT JOIN {store.get_data_source("chain_profile").get_table_query_string()} cp
            ON op.chain_id = cp.chain_id and op.country_code = cp.country_code and op.feature_timestamp = cp.feature_timestamp
            WHERE op.feature_timestamp BETWEEN "{start_date}" AND "{end_date}"
            AND op.country_code = "{country_code}"
            AND (vertical = 'food' or secondary_verticals like '%Food%')
        """

        data_df = store.get_historical_features(
            entity_df=entity_sql,
            features=[
                "order_profile_2t_v3_fv:delivery_area_id",
                "order_profile_2t_v3_fv:week_start",
                "order_profile_2t_v3_fv:geohash",
                "order_profile_2t_v3_fv:order_date",
                "order_profile_2t_v3_fv:order_time_utc",

                "account_orders_2t_v3_fv:most_recent_10_orders",  # user_prev_chains
                "account_orders_2t_v3_fv:frequent_chains",  # 'freq_chains'

                "account_search_2t_v3_fv:most_recent_15_search_keywords",  # prev_searches

                "account_performance_2t_v3_fv:account_log_order_cnt",
                "account_performance_2t_v3_fv:account_log_avg_gmv_eur",
                "account_performance_2t_v3_fv:account_incentives_pct",
                "account_performance_2t_v3_fv:account_is_tpro",
                "account_performance_2t_v3_fv:account_discovery_pct",

                "account_engagement_2t_v3_fv:most_recent_10_clicks_wo_orders",  # prev_clicks
                "account_engagement_2t_v3_fv:frequent_clicks",  # 'freq_clicks'
                "account_impressions_2t_v3_fv:frequent_neg_impressions"
            ],
        ).to_df()

        rename_dict = {
            "most_recent_10_orders": "prev_chains",
            "frequent_chains": "freq_chains",
            "most_recent_10_clicks_wo_orders": "prev_clicks",
            "frequent_clicks": "freq_clicks",
            "most_recent_15_search_keywords": "prev_searches",
        }

        data_df.rename(columns=rename_dict, inplace=True)

        #TODO: resolve the datetime type fixes
        data_df['order_hour'] = data_df['order_time_utc'].dt.hour
        data_df['order_weekday'] = data_df['order_time_utc'].dt.dayofweek
        # define a dict and map it bw 0-6
        data_df['geohash6'] = data_df['geohash'].astype(str).str[:6]

        data_df["account_log_avg_gmv_eur"] = data_df["account_log_avg_gmv_eur"].astype("float32")
        data_df['chain_id'] = data_df['chain_id'].astype(str)
        data_df["delivery_area_id"] = data_df["delivery_area_id"].astype("string")
        data_df['delivery_area_id'] = data_df['delivery_area_id'].fillna('asdasdas')

        # float_columns = data_df.select_dtypes(include=["float32", "float64"]).columns
        # data_df[float_columns] = data_df[float_columns].fillna(0)

        data_df['account_is_tpro'] = data_df['account_is_tpro'].fillna(0)
        data_df['account_discovery_pct'] = data_df['account_discovery_pct'].fillna(1)
        data_df['account_log_order_cnt'] = data_df['account_log_order_cnt'].fillna(0)
        data_df['account_log_avg_gmv_eur'] = data_df['account_log_avg_gmv_eur'].fillna(
            data_df['account_log_avg_gmv_eur'].mean())
        data_df['account_incentives_pct'] = data_df['account_incentives_pct'].fillna(
            data_df['account_incentives_pct'].mean())

        data_df['freq_chains'] = data_df['freq_chains'].fillna('no_frequent_orders')
        data_df['freq_clicks'] = data_df['freq_clicks'].fillna('no_frequent_clicks')
        data_df['prev_clicks'] = data_df['prev_clicks'].fillna('no_recent_clicks')
        data_df['prev_chains'] = data_df['prev_chains'].fillna('first_order')
        data_df['frequent_neg_impressions'] = data_df['frequent_neg_impressions'].fillna('no_negative_impressions')
        data_df['prev_searches'] = data_df['prev_searches'].fillna('no_prev_search')

        data_df = data_df.sort_values(by='order_time_utc')
        data_df = data_df.tail(data_points - 1)

        data_df.to_parquet(filename)

        upload_to_gcs(BUCKET_NAME, filename, blob)

    LOG.info(f"Start date: {data_df['order_date'].min()}")
    LOG.info(f"End date: {data_df['order_date'].max()}")
    LOG.info(f"Data Shape: {data_df.shape}")
    mlflow.log_param("train_start_date", data_df['order_date'].min())
    mlflow.log_param("train_end_date", data_df['order_date'].max())

    data_df.drop(columns=['country_code', 'event_timestamp', 'order_time_utc', 'order_date', 'week_start'],inplace=True)

    return data_df


def read_test_data_fs(start_date, end_date, country_code, data_points, is_val=False):
    if is_val:
        filename = 'raw_val_data_fs_' + country_code + '_' + end_date + '.parquet'
        data_str = 'val'
    else:
        filename = 'raw_test_data_fs_' + country_code + '_' + end_date + '.parquet'
        data_str = 'test'

    if os.path.exists(filename):
        data_df = pd.read_parquet(filename)
    else:
        entity_sql = f"""
            SELECT
                op.order_id,
                op.account_id,
                op.chain_id,
                op.country_code,
                op.feature_timestamp AS event_timestamp
            FROM {store.get_data_source("orders_profile").get_table_query_string()} op
            LEFT JOIN {store.get_data_source("chain_profile").get_table_query_string()} cp
            ON op.chain_id = cp.chain_id and op.country_code = cp.country_code and op.feature_timestamp = cp.feature_timestamp
            WHERE op.feature_timestamp BETWEEN "{start_date}" AND "{end_date}"
            AND op.country_code = "{country_code}"
            AND (vertical = 'food' or secondary_verticals like '%Food%')
        """

        data_df = store.get_historical_features(
            entity_df=entity_sql,
            features=[

                "order_profile_2t_v3_fv:delivery_area_id",
                "order_profile_2t_v3_fv:week_start",
                "order_profile_2t_v3_fv:geohash",
                # "order_profile_2t_v3_fv:order_gmv_eur",
                "order_profile_2t_v3_fv:order_date",
                "order_profile_2t_v3_fv:order_time_utc",

                "account_orders_2t_v3_fv:most_recent_10_orders",  # user_prev_chains
                "account_orders_2t_v3_fv:frequent_chains",  # 'freq_chains'

                "account_search_2t_v3_fv:most_recent_15_search_keywords",  # prev_searches

                "account_performance_2t_v3_fv:account_log_order_cnt",
                "account_performance_2t_v3_fv:account_log_avg_gmv_eur",
                "account_performance_2t_v3_fv:account_incentives_pct",
                "account_performance_2t_v3_fv:account_is_tpro",
                "account_performance_2t_v3_fv:account_discovery_pct",

                "account_engagement_2t_v3_fv:most_recent_10_clicks_wo_orders",  # prev_clicks
                "account_engagement_2t_v3_fv:frequent_clicks",  # 'freq_clicks'
                "account_impressions_2t_v3_fv:frequent_neg_impressions"
            ],
        ).to_df()

        rename_dict = {
            "most_recent_10_orders": "prev_chains",
            "frequent_chains": "freq_chains",
            "most_recent_10_clicks_wo_orders": "prev_clicks",
            "frequent_clicks": "freq_clicks",
            "most_recent_15_search_keywords": "prev_searches",
        }

        data_df.rename(columns=rename_dict, inplace=True)

        #TODO: resolve the datetime type fixes
        data_df['order_hour'] = data_df['order_time_utc'].dt.hour
        data_df['order_weekday'] = data_df['order_time_utc'].dt.dayofweek
        # define a dict and map it bw 0-6
        data_df['geohash6'] = data_df['geohash'].astype(str).str[:6]

        data_df["account_log_avg_gmv_eur"] = data_df["account_log_avg_gmv_eur"].astype("float32")
        data_df['chain_id'] = data_df['chain_id'].astype(str)
        data_df["delivery_area_id"] = data_df["delivery_area_id"].astype("string")
        data_df['delivery_area_id'] = data_df['delivery_area_id'].fillna('asdasdas')

        # float_columns = data_df.select_dtypes(include=["float32", "float64"]).columns
        # data_df[float_columns] = data_df[float_columns].fillna(0)

        data_df['account_is_tpro'] = data_df['account_is_tpro'].fillna(0)
        data_df['account_discovery_pct'] = data_df['account_discovery_pct'].fillna(1)
        data_df['account_log_order_cnt'] = data_df['account_log_order_cnt'].fillna(0)
        data_df['account_log_avg_gmv_eur'] = data_df['account_log_avg_gmv_eur'].fillna(
            data_df['account_log_avg_gmv_eur'].mean())
        data_df['account_incentives_pct'] = data_df['account_incentives_pct'].fillna(
            data_df['account_incentives_pct'].mean())

        data_df['freq_chains'] = data_df['freq_chains'].fillna('no_frequent_orders')
        data_df['freq_clicks'] = data_df['freq_clicks'].fillna('no_frequent_clicks')
        data_df['prev_clicks'] = data_df['prev_clicks'].fillna('no_recent_clicks')
        data_df['prev_chains'] = data_df['prev_chains'].fillna('first_order')
        data_df['frequent_neg_impressions'] = data_df['frequent_neg_impressions'].fillna('no_negative_impressions')
        data_df['prev_searches'] = data_df['prev_searches'].fillna('no_prev_search')


        data_df = data_df.groupby('order_date').apply(lambda x: x.sample(n=data_points, random_state=42)).reset_index(drop=True)

        data_df.to_parquet(filename)

    LOG.info(f"Start date: {data_df['order_date'].min()}")
    LOG.info(f"End date: {data_df['order_date'].max()}")
    LOG.info(f"Data Shape: {data_df.shape}")

    mlflow.log_param(f"{data_str}_start_date", data_df['order_date'].min())
    mlflow.log_param(f"{data_str}_end_date", data_df['order_date'].max())

    data_df.drop(columns=['country_code', 'event_timestamp', 'order_time_utc', 'order_date', 'week_start'],inplace=True)

    return data_df


def read_test_data_gcs_fs(start_date, end_date, country_code, data_points, is_val=False):
    if is_val:
        data_str = 'val'
    else:
        data_str = 'test'
    filename = f"raw_{data_str}_data_fs_" + country_code + '_' + start_date + '_' + end_date + '.parquet'
    blob = f"{DESTINATION_BLOB_NAME}/{country_code}/"

    if check_if_blob_exists(BUCKET_NAME, filename, blob):
        LOG.info(f"Data loading from {BUCKET_NAME}/{blob}/{filename}")
        data_df = read_parquet_from_gcs(BUCKET_NAME, filename, blob)

    else:
        entity_sql = f"""
            SELECT
                op.order_id,
                op.account_id,
                op.chain_id,
                op.country_code,
                op.feature_timestamp AS event_timestamp
            FROM {store.get_data_source("orders_profile").get_table_query_string()} op
            LEFT JOIN {store.get_data_source("chain_profile").get_table_query_string()} cp
            ON op.chain_id = cp.chain_id and op.country_code = cp.country_code and op.feature_timestamp = cp.feature_timestamp
            WHERE op.feature_timestamp BETWEEN "{start_date}" AND "{end_date}"
            AND op.country_code = "{country_code}"
            AND (vertical = 'food' or secondary_verticals like '%Food%')
        """

        data_df = store.get_historical_features(
            entity_df=entity_sql,
            features=[

                "order_profile_2t_v3_fv:delivery_area_id",
                "order_profile_2t_v3_fv:week_start",
                "order_profile_2t_v3_fv:geohash",
                # "order_profile_2t_v3_fv:order_gmv_eur",
                "order_profile_2t_v3_fv:order_date",
                "order_profile_2t_v3_fv:order_time_utc",

                "account_orders_2t_v3_fv:most_recent_10_orders",  # user_prev_chains
                "account_orders_2t_v3_fv:frequent_chains",  # 'freq_chains'

                "account_search_2t_v3_fv:most_recent_15_search_keywords",  # prev_searches

                "account_performance_2t_v3_fv:account_log_order_cnt",
                "account_performance_2t_v3_fv:account_log_avg_gmv_eur",
                "account_performance_2t_v3_fv:account_incentives_pct",
                "account_performance_2t_v3_fv:account_is_tpro",
                "account_performance_2t_v3_fv:account_discovery_pct",

                "account_engagement_2t_v3_fv:most_recent_10_clicks_wo_orders",  # prev_clicks
                "account_engagement_2t_v3_fv:frequent_clicks",  # 'freq_clicks'
                "account_impressions_2t_v3_fv:frequent_neg_impressions"
            ],
        ).to_df()

        rename_dict = {
            "most_recent_10_orders": "prev_chains",
            "frequent_chains": "freq_chains",
            "most_recent_10_clicks_wo_orders": "prev_clicks",
            "frequent_clicks": "freq_clicks",
            "most_recent_15_search_keywords": "prev_searches",
        }

        data_df.rename(columns=rename_dict, inplace=True)

        data_df['order_hour'] = data_df['order_time_utc'].dt.hour
        data_df['order_weekday'] = data_df['order_time_utc'].dt.dayofweek
        # define a dict and map it bw 0-6
        data_df['geohash6'] = data_df['geohash'].astype(str).str[:6]

        data_df["account_log_avg_gmv_eur"] = data_df["account_log_avg_gmv_eur"].astype("float32")
        data_df['chain_id'] = data_df['chain_id'].astype(str)
        data_df["delivery_area_id"] = data_df["delivery_area_id"].astype("string")
        data_df['delivery_area_id'] = data_df['delivery_area_id'].fillna('asdasdas')

        # float_columns = data_df.select_dtypes(include=["float32", "float64"]).columns
        # data_df[float_columns] = data_df[float_columns].fillna(0)

        data_df['account_is_tpro'] = data_df['account_is_tpro'].fillna(0)
        data_df['account_discovery_pct'] = data_df['account_discovery_pct'].fillna(1)
        data_df['account_log_order_cnt'] = data_df['account_log_order_cnt'].fillna(0)
        data_df['account_log_avg_gmv_eur'] = data_df['account_log_avg_gmv_eur'].fillna(
            data_df['account_log_avg_gmv_eur'].mean())
        data_df['account_incentives_pct'] = data_df['account_incentives_pct'].fillna(
            data_df['account_incentives_pct'].mean())

        data_df['freq_chains'] = data_df['freq_chains'].fillna('no_frequent_orders')
        data_df['freq_clicks'] = data_df['freq_clicks'].fillna('no_frequent_clicks')
        data_df['prev_clicks'] = data_df['prev_clicks'].fillna('no_recent_clicks')
        data_df['prev_chains'] = data_df['prev_chains'].fillna('first_order')
        data_df['frequent_neg_impressions'] = data_df['frequent_neg_impressions'].fillna('no_negative_impressions')
        data_df['prev_searches'] = data_df['prev_searches'].fillna('no_prev_search')


        data_df = data_df.groupby('order_date').apply(lambda x: x.sample(n=data_points, random_state=42)).reset_index(drop=True)

        data_df.to_parquet(filename)

        upload_to_gcs(BUCKET_NAME, filename, blob)

    LOG.info(f"Start date: {data_df['order_date'].min()}")
    LOG.info(f"End date: {data_df['order_date'].max()}")
    LOG.info(f"Data Shape: {data_df.shape}")

    mlflow.log_param(f"{data_str}_start_date", data_df['order_date'].min())
    mlflow.log_param(f"{data_str}_end_date", data_df['order_date'].max())

    data_df.drop(columns=['country_code', 'event_timestamp', 'order_time_utc', 'order_date', 'week_start'],inplace=True)

    return data_df


def read_data_inference_fs(start_date, end_date, country_code):

    entity_sql = f"""
        SELECT
            ao.account_id,
            ao.country_code,
            ao.feature_timestamp AS event_timestamp
        FROM {store.get_data_source("account_orders").get_table_query_string()} ao
        WHERE ao.feature_timestamp BETWEEN "{start_date}" AND "{end_date}"
        AND ao.country_code = "{country_code}"
    """

    data_df = store.get_historical_features(
        entity_df=entity_sql,
        features=[
            "account_performance_2t_v3_fv:account_log_order_cnt",
            "account_performance_2t_v3_fv:account_log_avg_gmv_eur",
            "account_performance_2t_v3_fv:account_incentives_pct",
            "account_performance_2t_v3_fv:account_is_tpro",
            "account_performance_2t_v3_fv:account_discovery_pct",

            "account_orders_2t_v3_fv:most_recent_10_orders",  # user_prev_chains
            "account_orders_2t_v3_fv:frequent_chains",  # 'freq_chains'

            "account_engagement_2t_v3_fv:most_recent_10_clicks_wo_orders",  # prev_clicks
            "account_engagement_2t_v3_fv:frequent_clicks",  # 'freq_clicks'
            "account_impressions_2t_v3_fv:frequent_neg_impressions"
        ],
    ).to_df()

    rename_dict = {
        "most_recent_10_orders": "prev_chains",
        "frequent_chains": "freq_chains",
        "most_recent_10_clicks_wo_orders": "prev_clicks",
        "frequent_clicks": "freq_clicks",
        "most_recent_15_search_keywords": "prev_searches",
    }
    data_df.rename(columns=rename_dict, inplace=True)

    data_df['account_is_tpro'] = data_df['account_is_tpro'].fillna(0)
    data_df['account_discovery_pct'] = data_df['account_discovery_pct'].fillna(1)
    data_df['account_log_order_cnt'] = data_df['account_log_order_cnt'].fillna(0)
    data_df['account_log_avg_gmv_eur'] = data_df['account_log_avg_gmv_eur'].fillna(
        data_df['account_log_avg_gmv_eur'].mean())
    data_df['account_incentives_pct'] = data_df['account_incentives_pct'].fillna(
        data_df['account_incentives_pct'].mean())

    data_df['freq_chains'] = data_df['freq_chains'].fillna('no_frequent_orders')
    data_df['freq_clicks'] = data_df['freq_clicks'].fillna('no_frequent_clicks')
    data_df['prev_clicks'] = data_df['prev_clicks'].fillna('no_recent_clicks')
    data_df['prev_chains'] = data_df['prev_chains'].fillna('first_order')
    data_df['frequent_neg_impressions'] = data_df['frequent_neg_impressions'].fillna('no_negative_impressions')

    return data_df

def read_data_active_inference_fs(start_date, end_date, country_code):

    entity_sql = f"""
        WITH recent_sessions AS (
            SELECT
                fs.account_id,
                fs.country_code
            FROM `tlb-data-prod.data_platform.fct_session` AS fs
            WHERE
                fs.date BETWEEN "{end_date}"-1 AND "{end_date}"
                AND fs.country_code = "{country_code}"
        ),
        recent_orders AS (
            SELECT
                foi.account_id,
                foi.country_code
            FROM `tlb-data-prod.data_platform.fct_order_info` AS foi
            WHERE
                foi.order_date BETWEEN "{end_date}"-1 AND "{end_date}"
                AND foi.country_code = "{country_code}"
        ),
        active_accounts AS (
            SELECT DISTINCT
                account_id,
                country_code
            FROM (
                SELECT account_id, country_code FROM recent_sessions
                UNION ALL
                SELECT account_id, country_code FROM recent_orders
            ) combined_data
        )
        SELECT
            ao.account_id,
            ao.country_code,
            ao.feature_timestamp AS event_timestamp
        FROM {store.get_data_source("account_orders").get_table_query_string()} ao
        JOIN active_accounts aa
            ON ao.account_id = aa.account_id
            AND ao.country_code = aa.country_code
        WHERE
            ao.feature_timestamp BETWEEN "{start_date}" AND "{end_date}"
            AND ao.country_code = "{country_code}"
    """

    data_df = store.get_historical_features(
        entity_df=entity_sql,
        features=[
            "account_performance_2t_v3_fv:account_log_order_cnt",
            "account_performance_2t_v3_fv:account_log_avg_gmv_eur",
            "account_performance_2t_v3_fv:account_incentives_pct",
            "account_performance_2t_v3_fv:account_is_tpro",
            "account_performance_2t_v3_fv:account_discovery_pct",
            "account_orders_2t_v3_fv:most_recent_10_orders",
            "account_orders_2t_v3_fv:frequent_chains",
            "account_engagement_2t_v3_fv:most_recent_10_clicks_wo_orders",
            "account_engagement_2t_v3_fv:frequent_clicks",
            "account_impressions_2t_v3_fv:frequent_neg_impressions"
        ],
    ).to_df()

    rename_dict = {
        "most_recent_10_orders": "prev_chains",
        "frequent_chains": "freq_chains",
        "most_recent_10_clicks_wo_orders": "prev_clicks",
        "frequent_clicks": "freq_clicks",
        "most_recent_15_search_keywords": "prev_searches",
    }
    data_df.rename(columns=rename_dict, inplace=True)

    data_df['account_is_tpro'] = data_df['account_is_tpro'].fillna(0)
    data_df['account_discovery_pct'] = data_df['account_discovery_pct'].fillna(1)
    data_df['account_log_order_cnt'] = data_df['account_log_order_cnt'].fillna(0)
    data_df['account_log_avg_gmv_eur'] = data_df['account_log_avg_gmv_eur'].fillna(
        data_df['account_log_avg_gmv_eur'].mean())
    data_df['account_incentives_pct'] = data_df['account_incentives_pct'].fillna(
        data_df['account_incentives_pct'].mean())

    data_df['freq_chains'] = data_df['freq_chains'].fillna('no_frequent_orders')
    data_df['freq_clicks'] = data_df['freq_clicks'].fillna('no_frequent_clicks')
    data_df['prev_clicks'] = data_df['prev_clicks'].fillna('no_recent_clicks')
    data_df['prev_chains'] = data_df['prev_chains'].fillna('first_order')
    data_df['frequent_neg_impressions'] = data_df['frequent_neg_impressions'].fillna('no_negative_impressions')

    return data_df


def read_data_inference_gf_fs(start_date, end_date, country_code):
    entity_sql = f"""
        SELECT
            agf.account_id,
            agf.country_code,
            agf.model_name,
            agf.feature_timestamp AS event_timestamp
        FROM {store.get_data_source("account_guest_features").get_table_query_string()} agf
        WHERE agf.feature_timestamp BETWEEN "{start_date}" AND "{end_date}"
        AND agf.country_code = "{country_code}"
    """

    data_df = store.get_historical_features(
        entity_df=entity_sql,
        features=[
            "account_guest_features_2t_v1_fv:account_log_order_cnt",
            "account_guest_features_2t_v1_fv:account_log_avg_gmv_eur",
            "account_guest_features_2t_v1_fv:account_incentives_pct",
            "account_guest_features_2t_v1_fv:account_is_tpro",
            "account_guest_features_2t_v1_fv:account_discovery_pct",
            "account_guest_features_2t_v1_fv:most_recent_10_orders",
            "account_guest_features_2t_v1_fv:frequent_chains",
            "account_guest_features_2t_v1_fv:most_recent_10_clicks_wo_orders",
            "account_guest_features_2t_v1_fv:frequent_clicks",
            "account_guest_features_2t_v1_fv:frequent_neg_impressions"
        ],
    ).to_df()

    rename_dict = {
        "most_recent_10_orders": "prev_chains",
        "frequent_chains": "freq_chains",
        "most_recent_10_clicks_wo_orders": "prev_clicks",
        "frequent_clicks": "freq_clicks",
        "most_recent_15_search_keywords": "prev_searches",
    }
    data_df.rename(columns=rename_dict, inplace=True)

    data_df.drop(columns=['model_name'], inplace=True)

    return data_df

def clean_test_data(test_data, model_artifacts):
    ## Clean based on geohash
    # Get the set of keys from the dictionary
    keys_set = set(model_artifacts['geohash_to_index'].keys())
    # Get the values from column 'A' that are not in the dictionary keys
    non_matching_values = test_data[~test_data['geohash6'].isin(keys_set)]
    # Count the number of rows with values not in the dictionary keys
    non_matching_count = non_matching_values.shape[0]
    print(f"Number of rows with values not corresponding to any key in geohash dict: {non_matching_count}")

    test_data = test_data[test_data['geohash6'].isin(keys_set)]

    ## Clean based on areaid
    # Get the set of keys from the dictionary
    keys_set = set(model_artifacts['area_id_to_index'].keys())
    # Get the values from column 'A' that are not in the dictionary keys
    non_matching_values = test_data[~test_data['delivery_area_id'].isin(keys_set)]
    # Count the number of rows with values not in the dictionary keys
    non_matching_count = non_matching_values.shape[0]
    print(f"Number of rows with values not corresponding to any key in delivery_area_id dict: {non_matching_count}")
    test_data = test_data[test_data['delivery_area_id'].isin(keys_set)]

    print(test_data.shape)

    # Get the vocabulary set from the tokenizer
    vocab_set = set(model_artifacts['chain_id_vocab'].keys())

    # Filter the DataFrame to find rows where the column 'A' values are not in the tokenizer vocabulary
    non_matching_values = test_data[~test_data['chain_id'].isin(vocab_set)]

    # Count the number of rows with values not in the tokenizer vocabulary
    non_matching_count = non_matching_values.shape[0]

    print(f"Number of rows with values not in the tokenizer vocabulary: {non_matching_count}")
    test_data = test_data[test_data['chain_id'].isin(vocab_set)]

    return test_data

def train_val_split_data(data, nmb_train_samples=1000000, nmb_val_samples=10000, nmb_train_val_samples=5120):
    train_data = data[:nmb_train_samples]
    train_val_data = train_data.sample(n=nmb_train_val_samples, random_state=42)  # random_state for reproducibility

    if nmb_train_samples + nmb_train_val_samples < data.shape[0]:
        val_data = data[nmb_train_samples:].sample(n=nmb_val_samples, random_state=42)
    else:
        return train_data, None, train_val_data

    return train_data, val_data, train_val_data

if __name__ == "__main__":
    start_date = "2024-09-12"
    end_date = "2024-09-12"
    country_code = "IQ"

    data_df = read_data_active_inference_fs(
        start_date=start_date,
        end_date=end_date,
        country_code=country_code
    )