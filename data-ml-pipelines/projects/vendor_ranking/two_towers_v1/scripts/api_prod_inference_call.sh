#!/usr/bin/env bash
set -x #echo on
curl -X POST "https://ace.talabat.com/models/${2:-vendor-ranking-tt-v1}" --fail -H "accept: application/json" \
     -H "Content-Type: application/json" -d '{"inputs": {
    "account_discovery_pct": 0.2,
    "account_incentives_pct": 0.03333333333333333,
    "account_is_tpro": 0.0,
    "account_log_avg_gmv_eur": 2.22116114,
    "account_log_order_cnt": 3.4339872044851463,
    "delivery_area_id": "8054",
    "freq_chains": "no_frequent_orders",
    "freq_clicks": "502350",
    "geohash6": "stq503",
    "order_hour": 15,
    "order_weekday": 3,
    "prev_clicks": "503616 602304 660700 618120 502350 660608 608498 505503 502313 607021",
    "user_prev_chains": "511030 606257 502321 511030 511030 511030 511030 502349 511030 511030",
    "prev_searches": "ابيتا ابو جباره لوكارب ماكدونلدز ستاربكس"
}}'

