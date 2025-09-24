import logging
from datetime import datetime
from typing import Sequence

import newrelic.agent
import polars as pl
import pytz
import structlog

import ace
from ace.perf import perf_manager
from ace.storage import db
from vendor_ranking.key_preferred_partner.logic import KeyPreferredPartner
from abstract_ranking.two_tower.artefacts_service import ACTIVE_TT_VERSIONS

logger = structlog.get_logger()

all_vendors: pl.DataFrame | None = None
vendor_penalty_scores: pl.DataFrame | None = None
ranking_penalties_matrix: pl.DataFrame | None = None
ranking_penalties_matrix_height: int | None = None
vendor_penalty_scores_v2: pl.DataFrame | None = None
ranking_penalties_matrix_v2: pl.DataFrame | None = None
ranking_penalties_matrix_height_v2: int | None = None
kpp_vendor_scores: pl.DataFrame | None = None
kpp_overall_avg_threshold: float | None = None

pp_v2_vendor_scores_table_name: str = "vl_ranking_penalties_v2"
pp_v2_ranking_penalties_matrix_table_name: str = "vendor_ranking_penalty_matrix_v2"


def is_loaded() -> bool:
    from vendor_ranking.context import ctx

    if not (context := ctx()) or not context.opened:
        logger.warning("[startup] Context is not opened")
        return False

    # Todo - check old pp logic tables needed - check kpp tables needed
    dfs = (
        (all_vendors, "vendors ranking features", True),
        (vendor_penalty_scores, "vendor penalty scores", False),
        (ranking_penalties_matrix, "ranking penalties matrix", True),
        (vendor_penalty_scores_v2, "vendor penalty scores (v2)", True),
        (ranking_penalties_matrix_v2, "ranking penalties matrix (v2)", True),
        (kpp_vendor_scores, "kpp vendor scores", True),
    )

    for df, df_name, must_be_full in dfs:
        if df is None:
            logger.warning(f"[startup] DF `{df_name}` is not loaded")
            return False
        if must_be_full and df.is_empty():
            logger.warning(f"[startup] DF `{df_name}` is empty")
            return False

    for version in ACTIVE_TT_VERSIONS:
        artifacts_service = context.artifacts_service_registry.artifacts_services.get(version)
        for country, df in artifacts_service.get_embeddings_per_country():
            if df is None:
                logger.warning(f"[startup] Chain embeddings is not loaded for {version} {country}")
                return False

    return True


VENDORS_QUERY = """SELECT
                    country_id,
                    date(dwh_entry_timestamp)  dwh_entry_date,
                    vendor_id,
                    is_tgo,
                    rating_count,
                    online_days,
                    popularity_org_score,
                    _1_midnight_snack_score,
                    _2_breakfast_score,
                    _3_lunch_score,
                    _4_evening_snack_score,
                    _5_dinner_score,
                    new_customer_retention_14,
                    vendor_fail_rate,
                    aov_eur,
                    new_order_prc,
                    promotion_order_prc,
                    menu_to_cart_rate,
                    dwell_time_seconds_mean,
                    chain_id,
                    chain_name,
                    COALESCE(embeddings, ARRAY_FILL(0, ARRAY[50])) AS embeddings
                FROM vl_feature_vendors_v3"""
if ace.DEBUG:
    VENDORS_QUERY += " LIMIT 100000"

VENDORS_SENSOR_QUERY = """SELECT CASE
    WHEN (SELECT MAX(date(dwh_entry_timestamp)) FROM vl_feature_vendors_v3) = $1 THEN FALSE
    ELSE TRUE END is_data_refreshed"""

VENDOR_PENALTY_SCORES_QUERY = "SELECT vendor_id, penalty_bin FROM vl_ranking_penalties"

RANKING_PENALTIES_MATRIX_QUERY = (
    "SELECT rank, low_score_penalty, medium_score_penalty, high_score_penalty FROM vl_ranking_penalties_matrix"
)

VENDOR_PENALTY_SCORES_V2_QUERY = f"SELECT vendor_id, penalty_bin FROM {pp_v2_vendor_scores_table_name}"

RANKING_PENALTIES_MATRIX_V2_QUERY = f"""SELECT
    rank, bin_1, bin_2, bin_3, bin_4, bin_5, bin_6, bin_7, bin_8, bin_9, bin_10 
    FROM {pp_v2_ranking_penalties_matrix_table_name}"""

KPP_VENDOR_SCORES_QUERY = """SELECT vendor_id, avg_score, stdev_score FROM kpp_champions_with_similarity_scores"""

score_cols_list = [
    "_1_midnight_snack_score",
    "_2_breakfast_score",
    "_3_lunch_score",
    "_4_evening_snack_score",
    "_5_dinner_score",
]


def select_score(hour: int) -> str:
    if 0 <= hour <= 6:
        return "_1_midnight_snack_score"
    elif 7 <= hour <= 11:
        return "_2_breakfast_score"
    elif 12 <= hour <= 15:
        return "_3_lunch_score"
    elif 16 <= hour <= 18:
        return "_4_evening_snack_score"
    elif 19 <= hour < 24:
        return "_5_dinner_score"


COUNTRY_TIMEZONE = {
    1: pytz.timezone("Asia/Kuwait"),
    2: pytz.timezone("Asia/Riyadh"),
    3: pytz.timezone("Asia/Bahrain"),
    4: pytz.timezone("Asia/Dubai"),
    5: pytz.timezone("Asia/Muscat"),
    6: pytz.timezone("Asia/Qatar"),
    8: pytz.timezone("Asia/Amman"),
    9: pytz.timezone("Africa/Cairo"),
    10: pytz.timezone("Asia/Baghdad"),
}


async def refresh_ranking_penalties():
    global vendor_penalty_scores
    global ranking_penalties_matrix
    global ranking_penalties_matrix_height

    logger.info("Getting ranking penalties..")
    async with db.connections().acquire() as conn:
        vendor_penalty_scores = await db.fetch_as_df(conn, VENDOR_PENALTY_SCORES_QUERY)
        ranking_penalties_matrix = await db.fetch_as_df(conn, RANKING_PENALTIES_MATRIX_QUERY)
        # TODO Change the type in the DB
        ranking_penalties_matrix = ranking_penalties_matrix.with_columns(
            pl.col("low_score_penalty").cast(pl.Int64),
            pl.col("medium_score_penalty").cast(pl.Int64),
            pl.col("high_score_penalty").cast(pl.Int64),
        )
        ranking_penalties_matrix_height = ranking_penalties_matrix.shape[0]  # TODO Use .count()?..

    logger.info(
        f"Ranking penalties has been loaded in memory: "
        f"ranking_penalties_matrix={ranking_penalties_matrix.estimated_size('mb'):.4f} mb, "
        f"vendor_penalty_scores={vendor_penalty_scores.estimated_size('mb'):.4f} mb"
    )


async def refresh_ranking_penalties_v2():
    global vendor_penalty_scores_v2
    global ranking_penalties_matrix_v2
    global ranking_penalties_matrix_height_v2

    logger.info("Getting ranking penalties..")
    async with db.connections().acquire() as conn:
        vendor_penalty_scores_v2 = await db.fetch_as_df(conn, VENDOR_PENALTY_SCORES_V2_QUERY)
        ranking_penalties_matrix_v2 = await db.fetch_as_df(conn, RANKING_PENALTIES_MATRIX_V2_QUERY)
        ranking_penalties_matrix_height_v2 = ranking_penalties_matrix_v2.shape[0]

    assert not vendor_penalty_scores_v2.is_empty(), "vendor penalty scores (v2) are not loaded at all"
    assert not ranking_penalties_matrix_v2.is_empty(), "penalty rank matrix (v2) is not loaded at all"

    logger.info(
        f"Ranking penalties (V2) has been loaded in memory: "
        f"ranking_penalties_matrix_v2={ranking_penalties_matrix_v2.estimated_size('mb'):.4f} mb, "
        f"vendor_penalty_scores_v2={vendor_penalty_scores_v2.estimated_size('mb'):.4f} mb"
    )


def get_pp_v2_scoring() -> tuple[pl.DataFrame, pl.DataFrame, int]:
    return vendor_penalty_scores_v2, ranking_penalties_matrix_v2, ranking_penalties_matrix_height_v2


async def refresh_kpp_scoring():
    global kpp_vendor_scores
    global kpp_overall_avg_threshold

    logger.info("Getting kpp vendor scores..")
    async with db.connections().acquire() as conn:
        kpp_vendor_scores = await db.fetch_as_df(conn, KPP_VENDOR_SCORES_QUERY)

    n = KeyPreferredPartner.STANDARD_DEVIATION_CONSTANT_OFFLINE
    kpp_vendor_scores = kpp_vendor_scores.with_columns(
        offline_average_threshold=pl.col("avg_score") + (n * pl.col("stdev_score"))
    )
    kpp_overall_avg_threshold = kpp_vendor_scores["offline_average_threshold"].mean()
    kpp_scores_null_count = kpp_vendor_scores["offline_average_threshold"].is_null().sum()
    kpp_scores_count = kpp_vendor_scores.shape[0]
    logger.info(
        f"KPP vendor scores (rows: {kpp_scores_count}) "
        f"has been loaded in memory {kpp_vendor_scores.estimated_size('mb'):.4f} mb"
    )
    logger.info(
        f"Offline std dev constant assigned: {n} - KPP table overall_avg_threshold computed: "
        f"{kpp_overall_avg_threshold} (from non null rows: {kpp_scores_count - kpp_scores_null_count})"
    )


async def refresh_vendors():
    global all_vendors

    data_status = {"is_data_refreshed": True}

    if all_vendors is not None:
        max_date = all_vendors["dwh_entry_date"].max()
        logger.info(f"Check if latest dynamic ranking vendor data is refreshed after {max_date}...")
        async with db.connections().acquire() as conn:
            data_status = await db.fetch_row_as_dict(conn, VENDORS_SENSOR_QUERY, max_date)
    if not data_status.get("is_data_refreshed"):
        logger.info("Skipping dynamic ranking data refresh...")
        return

    with perf_manager(
        description_before="Getting static vendors features...",
        description="Loaded static vendors features",
        level=logging.INFO,
    ):
        async with db.connections().acquire() as conn:
            all_vendors = await db.fetch_as_df(conn, VENDORS_QUERY)

    logger.info(f"Ranking static vendors features has been loaded in memory {all_vendors.estimated_size('mb'):.4f} mb")


@newrelic.agent.function_trace()
def get_vendors(country_id: int, vendors: Sequence[int] | pl.Series, timestamp: datetime) -> pl.LazyFrame:
    now = timestamp.astimezone(COUNTRY_TIMEZONE.get(country_id, pytz.utc))
    current_score_col = select_score(now.hour)

    # TODO Split by country to a dictionary
    df = (
        all_vendors.lazy()
        .filter((pl.col("country_id") == country_id) & pl.col("vendor_id").is_in(vendors))
        .with_columns(pl.col(current_score_col).alias("timeofday_score"))
        .drop(score_cols_list)
    )

    return df


def get_all_vendors() -> "pl.DataFrame":
    return all_vendors
