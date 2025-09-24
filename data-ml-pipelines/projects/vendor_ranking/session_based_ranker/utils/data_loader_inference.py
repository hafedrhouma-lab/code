from datetime import datetime
import os
import pandas as pd
from projects.vendor_ranking.session_based_ranker.utils.tutils_utils import read_query

def read_offline_inference_data(country_code):
    filename = f'inference_data_{country_code}_{datetime.today().strftime("%Y-%m-%d")}_data.parquet'
    if os.path.exists(filename):
        return pd.read_parquet(filename)

    q = f"""
        SELECT *
        FROM `tlb-data-prod.data_platform_personalization.inference_features_for_two_tower_v3_per_account` 
        WHERE country_code = '{country_code}'
    """

    accounts_df = read_query(q)

    accounts_df = accounts_df.rename(columns={
        'most_recent_10_orders': 'prev_chains',
        'frequent_chains': 'freq_chains',
        'frequent_clicks': 'freq_clicks',
        'most_recent_10_clicks_wo_orders': 'prev_clicks'
    })

    accounts_df.to_parquet(filename)
    return accounts_df


def get_active_accounts(country_code):
    q = f"""
        WITH recent_sessions AS (
        SELECT
            fs.account_id,
            fs.country_code
        FROM `tlb-data-prod.data_platform.fct_session` AS fs
        WHERE
            fs.account_id > 0
            AND fs.date = current_date() - 1
        ),
        recent_orders AS (
            SELECT
                foi.account_id,
                foi.country_code
            FROM `tlb-data-prod.data_platform.fct_order_info` AS foi
            WHERE
                foi.account_id > 0
                AND foi.order_date = current_date() - 1
        )
        SELECT DISTINCT 
            account_id, 
            country_code 
        FROM (
            SELECT account_id, country_code FROM recent_sessions
            UNION ALL
            SELECT account_id, country_code FROM recent_orders
        ) combined_data
        """

    active_accounts_df = read_query(q)
    active_account_ids = active_accounts_df['account_id'].tolist()

    return active_account_ids

