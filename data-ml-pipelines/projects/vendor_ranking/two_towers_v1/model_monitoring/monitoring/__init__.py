from .data_drift import (
    DataDriftDetector
)
from .target_drift import target_drift_test
from .model_metrics import MetricProcessor

USER_FEATURES_NON_TEXT = [
    "geohash6",
    "account_log_avg_gmv_eur",
    "delivery_area_id",
    "account_incentives_pct",
    "account_log_order_cnt",
    "order_hour",
    "account_discovery_pct",
    "account_is_tpro",
    "sample_weight"
]
USER_FEATURES_TEXT = [
    "freq_chains",
    "freq_clicks",
    "user_prev_chains",
    "prev_clicks"
]

CHAIN_FEATURES_NON_TEXT = [
    "chain_actual_tpro_orders_pct",
    "chain_avg_daily_active_orders_log",
    "chain_avg_delivery_fee_eur",
    "chain_avg_incentive_pct",
    "chain_avg_rating",
    "chain_contact_pct",
    "chain_daily_orders_per_vendor_log",
    "chain_fail_pct",
    "chain_hour_orders_pct_0",
    "chain_hour_orders_pct_1",
    "chain_hour_orders_pct_10",
    "chain_hour_orders_pct_11",
    "chain_hour_orders_pct_12",
    "chain_hour_orders_pct_13",
    "chain_hour_orders_pct_14",
    "chain_hour_orders_pct_15",
    "chain_hour_orders_pct_16",
    "chain_hour_orders_pct_17",
    "chain_hour_orders_pct_18",
    "chain_hour_orders_pct_19",
    "chain_hour_orders_pct_2",
    "chain_hour_orders_pct_20",
    "chain_hour_orders_pct_21",
    "chain_hour_orders_pct_22",
    "chain_hour_orders_pct_23",
    "chain_hour_orders_pct_3",
    "chain_hour_orders_pct_4",
    "chain_hour_orders_pct_5",
    "chain_hour_orders_pct_6",
    "chain_hour_orders_pct_7",
    "chain_hour_orders_pct_8",
    "chain_hour_orders_pct_9",
    "chain_incentivized_orders_pct",
    "chain_log_avg_gmv_eur",
    "chain_menu_desc_coverage_pct",
    "chain_menu_image_coverage_pct",
    "chain_menu_to_cart_pct",
    "chain_menu_to_order_pct",
    "chain_new_order_pct",
    "chain_potential_tpro_orders_pct",
    "chain_tgo_orders_pct",
]

CHAIN_FEATURES_TEXT = [
    "chain_freq_area_ids",
    "tlabel",
]

