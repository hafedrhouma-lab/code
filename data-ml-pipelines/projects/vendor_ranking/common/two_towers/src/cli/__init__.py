from typing import Iterable

CHAIN_MODEL_INPUT = {
    "chain_actual_tpro_orders_pct": 0.0,
    "chain_avg_daily_active_orders_log": 5.646192595199329,
    "chain_avg_delivery_fee_eur": 0.235974524,
    "chain_avg_incentive_pct": 0.150559223,
    "chain_avg_rating": 4.638562022649457,
    "chain_contact_pct": 0.004110362346963977,
    "chain_daily_orders_per_vendor_log": 3.3748914313049716,
    "chain_fail_pct": 0.013158910241568265,
    "chain_freq_area_ids": "7566 7600 7925 7612 7555 7616 7839 8055 7939 7628",
    "chain_hour_orders_pct_0": 0.0,
    "chain_hour_orders_pct_1": 0.0,
    "chain_hour_orders_pct_10": 0.020969520359139956,
    "chain_hour_orders_pct_11": 0.036465306765377646,
    "chain_hour_orders_pct_12": 0.04219500669449476,
    "chain_hour_orders_pct_13": 0.05032684886193589,
    "chain_hour_orders_pct_14": 0.06200283531542884,
    "chain_hour_orders_pct_15": 0.06979995274474285,
    "chain_hour_orders_pct_16": 0.08980467827045759,
    "chain_hour_orders_pct_17": 0.12924312829802315,
    "chain_hour_orders_pct_18": 0.14905095691895723,
    "chain_hour_orders_pct_19": 0.15029140741907537,
    "chain_hour_orders_pct_2": 0.0,
    "chain_hour_orders_pct_20": 0.11500748208238167,
    "chain_hour_orders_pct_21": 0.0380995510750571,
    "chain_hour_orders_pct_22": 0.0027565566669291957,
    "chain_hour_orders_pct_23": 0.0007482082381664961,
    "chain_hour_orders_pct_3": 0.0,
    "chain_hour_orders_pct_4": 0.0,
    "chain_hour_orders_pct_5": 0.0,
    "chain_hour_orders_pct_6": 0.0018311412144601086,
    "chain_hour_orders_pct_7": 0.0070882885721036465,
    "chain_hour_orders_pct_8": 0.015732062691974483,
    "chain_hour_orders_pct_9": 0.018587067811294006,
    "chain_id": "611536",
    "chain_incentivized_orders_pct": 0.1662794360872647,
    "chain_log_avg_gmv_eur": 1.914929224,
    "chain_menu_desc_coverage_pct": 0.011612473288833084,
    "chain_menu_image_coverage_pct": 1.0,
    "chain_menu_to_cart_pct": 0.39307809967353835,
    "chain_menu_to_order_pct": 0.23193845170422117,
    "chain_name": "ZAIN",
    "chain_new_order_pct": 0.2397141619748809,
    "chain_potential_tpro_orders_pct": 0.0,
    "chain_tgo_orders_pct": 1.0,
    "tlabel": "fruits"
}

USER_MODEL_INPUT = {
    "prev_clicks": "503616 602304 660700 618120 502350 660608 608498 505503 502313 607021",  # most_recent_10_clicks_wo_orders
    "freq_clicks": "502350",  # frequent_clicks
    "user_prev_chains": "511030 606257 502321 511030 511030 511030 511030 502349 511030 511030",  # most_recent_10_orders
    "most_recent_20_orders": "no_recent_orders",  # most_recent_20_orders
    "freq_chains": "no_frequent_orders",
    "account_order_source": "vendor_list",
    "account_log_order_cnt": 3.4339872044851463,
    "account_log_avg_gmv_eur": 2.22116114,
    "account_incentives_pct": 0.03333333333333333,
    "account_is_tpro": 0.0,
    "account_discovery_pct": 0.2,
    "prev_searches": "ابيتا ابو جباره لوكارب ماكدونلدز ستاربكس",
    "delivery_area_id": "8054",
    "geohash6": "stq503",
    "order_hour": 15,
    "order_weekday": 3,
}


WARMUP_USER_MODEL_INPUT_2 = {
    "prev_clicks": "503616 602304 660700 618120 502350 660608 608498 505503 502313 607021",
    "freq_clicks": "502350",  # frequent_clicks
    "user_prev_chains": "511030 606257 502321 511030 511030 511030 511030 502349 511030 511030",
    "most_recent_15_orders": "no_recent_orders",
    "most_recent_15_clicks_wo_orders": "503616 602304 660700",
    "freq_chains": "no_frequent_orders",
    "account_order_source": "vendor_list",
    "account_log_order_cnt": 3.4339872044851463,
    "account_log_avg_gmv_eur": 2.22116114,
    "account_incentives_pct": 0.03333333333333333,
    "account_is_tpro": 0.0,
    "account_discovery_pct": 0.2,
    "prev_searches": "ابيتا ابو جباره لوكارب ماكدونلدز ستاربكس",
    "delivery_area_id": "8054",
    "geohash6": "stq503",
    "order_hour": 15,
    "order_weekday": 3,
}

WARMUP_USER_MODEL_INPUT = {
    "prev_clicks": "503616 602304 660700 618120 502350 660608 608498 505503 502313 607021",
    "freq_clicks": "502350",
    "user_prev_chains": "511030 606257 502321 511030 511030 511030 511030 502349 511030 511030",
    "freq_chains": "no_frequent_orders",
    "account_log_order_cnt": 3.4339872044851463,
    "account_log_avg_gmv_eur": 2.22116114,
    "account_incentives_pct": 0.03333333333333333,
    "account_is_tpro": 0.0,
    "account_discovery_pct": 0.2,
    "delivery_area_id": "8054",
    "geohash6": "stq503",
    "order_hour": 15,
    "order_weekday": 3,
}

NAMES_MAPPING = {
    "most_recent_10_clicks_wo_orders": "prev_clicks",
    "frequent_clicks": "freq_clicks",
    "most_recent_10_orders": "user_prev_chains",
    "frequent_chains": "freq_chains",
    "most_recent_15_search_keywords": "prev_searches"
}


# account_guests_v1_all_fs
GUEST_USER_RAW_STATIC_FEATURES_NAMES: set[str] = {
    "most_recent_10_clicks_wo_orders", "frequent_clicks", "most_recent_10_orders",
    "frequent_chains", "account_log_order_cnt", "account_log_avg_gmv_eur",
    "account_incentives_pct", "account_is_tpro", "account_discovery_pct"
}

GUEST_USER_RAW_DYNAMIC_FEATURES_NAMES: set[str] = {
    "delivery_area_id", "order_weekday", "order_hour", "geohash6"
}

GUEST_USER_FEATURES_NAMES: set[str] = {
    NAMES_MAPPING.get(feature_name, feature_name)
    for feature_name in GUEST_USER_RAW_DYNAMIC_FEATURES_NAMES | GUEST_USER_RAW_STATIC_FEATURES_NAMES
}


# account_guests_v1_all_fs
BASE_USER_RAW_STATIC_FEATURES_NAMES: set[str] = {
    "most_recent_10_clicks_wo_orders", "frequent_clicks", "most_recent_10_orders",
    "frequent_chains", "account_log_order_cnt", "account_log_avg_gmv_eur",
    "account_incentives_pct", "account_is_tpro", "account_discovery_pct"
}


BASE_USER_RAW_DYNAMIC_FEATURES_NAMES: set[str] = {
    "delivery_area_id", "order_weekday", "order_hour", "geohash6"
}


BASE_USER_FEATURES_NAMES: set[str] = {
    NAMES_MAPPING.get(feature_name, feature_name)
    for feature_name in BASE_USER_RAW_DYNAMIC_FEATURES_NAMES | BASE_USER_RAW_STATIC_FEATURES_NAMES
}


def validate_features_names(current: Iterable[str], expected: Iterable[str]) -> None:
    current = set(current)
    expected = set(expected)
    lost_names = current.difference(expected)
    assert current == expected, f"Lost names: {lost_names}. Expected features names: {expected}, but got: {current}"
